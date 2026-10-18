# -*- coding: utf-8 -*-
"""日内降频特征：5分钟K线 → 日频（custom_intraday 家族，6个因子）。

数据源
------
`local_data/minute/<YYYYMM>/minute_data.parquet`（**不复权**；2020-01~2026-09，
81 个月；`minute_adj` 复权版仅作兜底）。
  列：stock_code("000001.SZ")/datetime/open/high/low/close/volume/amount。
  5分钟 bar（标签=区间终点：09:35~11:30 + 13:05~15:00，48 根/交易日）。

为何用不复权数据：本族 6 个因子全部是**日内同日内比值**（区间位置/同日收益/
同日成交占比/同日波动比/同日回撤），对"整日复权常数"不变——复权因子只在
日界变化（除权发生在日间开盘前），故日内口径下复权与否逐值等价；用不复权
可多吃 2.9 年覆盖（2020-01 起 vs 复权版 2023-06 起）。

日频聚合缓存
------------
`local_data/minute/daily_features_cache.parquet`（首次求值自动构建；
`python -m custom_factors.intraday_minute --build [--force]` 手动重建）。
新月份数据出现时自动检测并重建（meta 记录已并入月份清单）。

因子（全部只用当日 ≤15:00 的 bar——与日线特征同口径，无前视）
--------------------------------------------------------------
① `id_hi_amt_share20`   高位成交占比：收盘价位于当日 [low, high] 前 20% 区间的
                        bar 成交额 / 全天成交额（>50% 且收上影 = 高位派发嫌疑；
                        一字板/零区间日 NaN）
② `id_hi_shadow_dump`   高位派发强度：① × 日上影比例（铁血出货交互项）
③ `id_late30_ret`       尾盘30分钟动量：close(15:00)/close(14:30) − 1（T+1 抢筹）
④ `id_late30_amt_share` 尾盘30分钟成交占比：amount(14:35~15:00)/全天 amount
⑤ `id_updown_vol_ratio` 日内上下行波动比：std(阳线5分钟收益)/std(阴线5分钟收益)
⑥ `id_dd_recovery`      日内最大回撤修复度：(日终close − 回撤谷底)/(回撤峰 − 谷底)；
                        峰值/谷底取 bar high/low（防漏 bar 内高低点）；相对回撤
                        深度 <0.5% 视为无有效回撤=1.0；截断 [0,1]（防小回撤爆破，
                        旧口径实测 max=707）
⑦~⑩ `*_s3` 平滑变体     ① ② ⑤ ⑥ 的 3 日滚动均值（min_periods=2；覆盖月界；
                        降单日脉冲噪声与换手，原版保持裸值对照）

质量门槛：当日 bar 数 ≥20 才出值；⑤ 另要求阳线/阴线各 ≥3 根。
并发安全：首次构建经 flock 互斥 + 双检（多进程首用不重复建/不写坏）；
落盘为临时文件 + 原子 rename。多进程场景仍建议先手动 `--build` 一次。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(_HERE)
if _MERGE not in sys.path:
    sys.path.insert(0, _MERGE)

from custom_registry import register_py_factor  # noqa: E402

ROOT = Path(_MERGE).parent.parent  # integrated_stock_selection
_TO = ROOT.parent                  # .../to（local_data 的正式位置，gates.py 同约定）


def _resolve_min_dir() -> Path:
    """分钟数据目录解析：优先 to/local_data/minute（**不复权**，2020-01 起
    覆盖最全；日内比值因子对复权不变，见模块 docstring），次选
    integrated_stock_selection/local_data/minute；再兜底 minute_adj（复权版）。
    目录有效性 = "含至少一个月度 minute_data.parquet" **或** "含已构建的
    daily_features_cache.parquet"（cache-only 部署：服务器免传 7GB 原始分钟，
    仅上传缓存+meta 即可供因子求值）。"""
    for cand in (_TO / 'local_data' / 'minute',
                 ROOT / 'local_data' / 'minute',
                 _TO / 'local_data' / 'minute_adj',
                 ROOT / 'local_data' / 'minute_adj'):
        try:
            if not cand.exists():
                continue
            if (cand / 'daily_features_cache.parquet').exists():
                return cand
            if any(d.is_dir() and (d / 'minute_data.parquet').exists()
                   for d in cand.iterdir()):
                return cand
        except Exception:
            continue
    return _TO / 'local_data' / 'minute'


MIN_DIR = _resolve_min_dir()
CACHE_PATH = MIN_DIR / 'daily_features_cache.parquet'
META_PATH = MIN_DIR / 'daily_features_cache.meta.json'

FACTOR_COLS = ('hi_amt_share20', 'hi_shadow_dump', 'late30_ret',
               'late30_amt_share', 'updown_vol_ratio', 'dd_recovery',
               'hi_amt_share20_s3', 'hi_shadow_dump_s3',
               'updown_vol_ratio_s3', 'dd_recovery_s3')
#: 3日滚动均值平滑变体（降单日脉冲噪声/换手；覆盖月度边界，全量拼接后算）
SMOOTH3_BASE = ('hi_amt_share20', 'hi_shadow_dump', 'updown_vol_ratio',
                'dd_recovery')
_MIN_BARS = 20
_MIN_HALF = 3


def _month_files() -> list[Path]:
    if not MIN_DIR.exists():
        return []
    out = []
    for d in sorted(MIN_DIR.iterdir()):
        if d.is_dir() and (d / 'minute_data.parquet').exists():
            out.append(d / 'minute_data.parquet')
    return out


def _agg_month(fp: Path) -> pd.DataFrame | None:
    """单月 5 分钟 → 日频聚合（向量化；列名=内部原始量，factor 化在总装阶段）。"""
    cols = ['stock_code', 'datetime', 'high', 'low', 'close', 'amount']
    df = pd.read_parquet(fp, columns=cols)
    if not len(df):
        return None
    df['date'] = df['datetime'].dt.normalize()
    df = df.sort_values(['stock_code', 'date', 'datetime'],
                        kind='stable').reset_index(drop=True)
    code, codes_u = pd.factorize(df['stock_code'], sort=False)
    day, dates_u = pd.factorize(df['date'], sort=False)
    df['_code'], df['_day'] = code, day
    gk = ['_code', '_day']
    dt = df['datetime'].dt
    mins = (dt.hour.to_numpy() * 60 + dt.minute.to_numpy())
    close = df['close'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    amt = df['amount'].to_numpy(dtype=float)
    g = df.groupby(gk, sort=False)
    # 5分钟收益（组内 shift；prev<=0 置 NaN）
    prev = g['close'].shift(1).to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = close / np.where(prev > 0, prev, np.nan) - 1.0
    df['_r_up'] = np.where(r > 0, r, np.nan)
    df['_r_dn'] = np.where(r < 0, r, np.nan)
    # 日内回撤（组内 cummax；峰值用 bar high、谷底用 bar low，防漏 5 分钟内高低点）
    cm = g['high'].cummax().to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        dd = (cm - low) / np.where(cm > 0, cm, np.nan)
    df['_dd'] = dd
    dd_max = df.groupby(gk, sort=False)['_dd'].transform('max').to_numpy(float)
    trough = dd >= (dd_max - 1e-15)          # 首次触底（agg first 取最早行）
    df['_peak_t'] = np.where(trough, cm, np.nan)
    df['_trough_c'] = np.where(trough, low, np.nan)
    # 14:30 收盘（bar 标签 14:30 覆盖 14:25~14:30）与尾盘成交（14:35~15:00）
    df['_c1430'] = np.where(mins == 870, close, np.nan)
    df['_late_amt'] = np.where(mins > 870, amt, 0.0)
    # 当日区间与高位成交
    day_hi = df.groupby(gk, sort=False)['high'].transform('max').to_numpy(float)
    day_lo = df.groupby(gk, sort=False)['low'].transform('min').to_numpy(float)
    rng = day_hi - day_lo
    with np.errstate(invalid='ignore', divide='ignore'):
        pos = (close - day_lo) / np.where(rng > 0, rng, np.nan)
    df['_hi_amt'] = np.where(pos >= 0.8, amt, 0.0)
    agg = df.groupby(gk, sort=False).agg(
        amt_day=('amount', 'sum'),
        hi_amt=('_hi_amt', 'sum'),
        late_amt=('_late_amt', 'sum'),
        close_last=('close', 'last'),
        close_1430=('_c1430', 'first'),
        day_hi=('high', 'max'),
        day_lo=('low', 'min'),
        up_std=('_r_up', 'std'),
        dn_std=('_r_dn', 'std'),
        up_n=('_r_up', 'count'),
        dn_n=('_r_dn', 'count'),
        peak_t=('_peak_t', 'first'),
        trough_c=('_trough_c', 'first'),
        n_bars=('close', 'size'),
    ).reset_index()
    agg['stock_code'] = codes_u[agg['_code'].to_numpy()]
    agg['date'] = dates_u[agg['_day'].to_numpy()]
    return agg


def _finalize(agg: pd.DataFrame) -> pd.DataFrame:
    """原始聚合量 → 6 个因子值（含质量门槛）。"""
    out = agg[['stock_code', 'date']].copy()
    amt_day = agg['amt_day'].to_numpy(dtype=float)
    amt_ok = amt_day > 0
    rng = (agg['day_hi'] - agg['day_lo']).to_numpy(dtype=float)
    close_last = agg['close_last'].to_numpy(dtype=float)
    # ① 高位成交占比（零区间日/无成交日 NaN）
    with np.errstate(invalid='ignore', divide='ignore'):
        hi_share = np.where(amt_ok & (rng > 0),
                            agg['hi_amt'].to_numpy(float) / np.where(amt_ok, amt_day, np.nan),
                            np.nan)
    # ② 上影比例 × ①（交互：铁血出货强度）
    with np.errstate(invalid='ignore', divide='ignore'):
        shadow = np.where(rng > 0, (agg['day_hi'].to_numpy(float) - close_last)
                          / np.where(rng > 0, rng, np.nan), np.nan)
    dump = hi_share * shadow
    # ③ 尾盘30分钟动量
    c1430 = agg['close_1430'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        late_ret = close_last / np.where(c1430 > 0, c1430, np.nan) - 1.0
    # ④ 尾盘成交占比
    with np.errstate(invalid='ignore', divide='ignore'):
        late_share = np.where(amt_ok,
                              agg['late_amt'].to_numpy(float) / np.where(amt_ok, amt_day, np.nan),
                              np.nan)
    # ⑤ 上下行波动比（阴阳线各≥3根；分母下限1e-4防除爆，尾部由下游MAD软压缩兜底）
    up_s = agg['up_std'].to_numpy(dtype=float)
    dn_s = agg['dn_std'].to_numpy(dtype=float)
    ok5 = (agg['up_n'].to_numpy() >= _MIN_HALF) & (agg['dn_n'].to_numpy() >= _MIN_HALF) \
        & (dn_s > 1e-12)
    with np.errstate(invalid='ignore', divide='ignore'):
        updown = np.where(ok5, up_s / np.maximum(dn_s, 1e-4), np.nan)
    # ⑥ 最大回撤修复度（V4.3x 修复：小回撤防爆 + [0,1] 截断）
    # 旧口径对"回撤深度≈0"的噪声日会算出 +50~+700 的爆破值（实测max=707）；
    # 现：相对回撤深度 <0.5% 视为无有效回撤 → 1.0；有有效回撤才计算，
    # 并截断到 [0,1]（1.0=已修复至回撤前峰值；超出部分由动量族因子表达）。
    peak_t = agg['peak_t'].to_numpy(dtype=float)
    trough_c = agg['trough_c'].to_numpy(dtype=float)
    denom = peak_t - trough_c
    with np.errstate(invalid='ignore', divide='ignore'):
        depth_rel = np.where(peak_t > 0, denom / np.where(peak_t > 0, peak_t, np.nan), np.nan)
        _rec = (close_last - trough_c) / np.where(denom > 1e-12, denom, np.nan)
    rec = np.where(np.isfinite(depth_rel) & (depth_rel >= 0.005),
                   np.clip(_rec, 0.0, 1.0),
                   np.where(np.isfinite(trough_c), 1.0, np.nan))
    q = agg['n_bars'].to_numpy() >= _MIN_BARS
    for name, v in (('hi_amt_share20', hi_share), ('hi_shadow_dump', dump),
                    ('late30_ret', late_ret), ('late30_amt_share', late_share),
                    ('updown_vol_ratio', updown), ('dd_recovery', rec)):
        out[name] = np.where(q, v, np.nan).astype(np.float64)
    out['n_bars'] = agg['n_bars'].to_numpy().astype(np.int16)
    return out


def _cache_is_current() -> bool:
    """缓存是否与当前月份集一致（双检锁内复用）。
    cache-only 部署（无月度原始文件）：cache+meta 存在即视为最新（不触发
    需要原始数据的重建）。"""
    if not CACHE_PATH.exists() or not META_PATH.exists():
        return False
    try:
        meta = json.loads(META_PATH.read_text(encoding='utf-8'))
        months_now = [f.parent.name for f in _month_files()]
        if not months_now:
            return bool(meta.get('months'))
        return meta.get('months') == months_now
    except Exception:
        return False


def _build_with_lock() -> None:
    """跨进程互斥构建（flock；多进程首用并发安全）。
    拿锁后二次检查——等待期间别人已建好则直接跳过，不重复 8 份计算。
    Windows 无 fcntl：退化为无锁（本地单进程约定），仍靠原子写防损坏。"""
    lockp = CACHE_PATH.with_name(CACHE_PATH.name + '.lock')
    try:
        lf = open(lockp, 'w')
    except Exception:
        lf = None
    try:
        if lf is not None:
            try:
                import fcntl
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        if not _cache_is_current():
            build_cache(force=True)
    finally:
        if lf is not None:
            try:
                lf.close()
            except Exception:
                pass


def build_cache(force: bool = False) -> Path:
    """全量重建日频聚合缓存（逐月聚合→总装→s3平滑→原子落盘+meta）。"""
    files = _month_files()
    if not files:
        raise SystemExit(f"分钟数据目录为空：{MIN_DIR}")
    months = [f.parent.name for f in files]
    if not force and CACHE_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding='utf-8'))
            if meta.get('months') == months:
                print(f"cache 已是最新（{len(months)} 个月），跳过（--force 强制重建）",
                      flush=True)
                return CACHE_PATH
        except Exception:
            pass
    t0 = time.time()
    parts = []
    for i, fp in enumerate(files, 1):
        agg = _agg_month(fp)
        if agg is None:
            continue
        parts.append(agg)
        print(f"  [{i}/{len(files)}] {fp.parent.name} rows={len(agg):,} "
              f"{time.time() - t0:.0f}s", flush=True)
    full = pd.concat(parts, ignore_index=True)
    out = _finalize(full)
    # 3日滚动均值平滑变体（全量拼接后算，覆盖月度边界；min_periods=2）
    out = out.sort_values(['stock_code', 'date'], kind='stable').reset_index(drop=True)
    for _c in SMOOTH3_BASE:
        out[_c + '_s3'] = (out.groupby('stock_code', sort=False)[_c]
                           .rolling(3, min_periods=2).mean()
                           .reset_index(level=0, drop=True).to_numpy(dtype=np.float64))
    MIN_DIR.mkdir(parents=True, exist_ok=True)
    # 原子写：临时文件 + os.replace（并发下不产生半写文件）
    tmp_path = CACHE_PATH.with_name(CACHE_PATH.name + '.tmp')
    out.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, CACHE_PATH)
    META_PATH.write_text(json.dumps(
        {'months': months, 'rows': int(len(out)),
         'date_min': str(pd.to_datetime(out['date']).min().date()),
         'date_max': str(pd.to_datetime(out['date']).max().date()),
         'built_at': pd.Timestamp.now().isoformat(timespec='seconds')},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"cache -> {CACHE_PATH} rows={len(out):,} "
          f"({pd.to_datetime(out['date']).min().date()}~"
          f"{pd.to_datetime(out['date']).max().date()}) {time.time() - t0:.0f}s",
          flush=True)
    return CACHE_PATH


_CACHE_DF: pd.DataFrame | None = None
_CACHE_SERIES: dict[str, pd.Series] = {}
_WARNED = False


def _load_cache() -> pd.DataFrame | None:
    """惰性加载（月份集变化自动重建）；数据缺失返回 None（因子全 NaN）。"""
    global _CACHE_DF, _CACHE_SERIES, _WARNED
    if _CACHE_DF is not None:
        return _CACHE_DF
    if not MIN_DIR.exists():
        if not _WARNED:
            print(f"warning: 分钟数据目录缺失 {MIN_DIR}，intraday 因子全 NaN",
                  flush=True)
            _WARNED = True
        return None
    try:
        if not _cache_is_current():
            months = [f.parent.name for f in _month_files()]
            print(f"intraday: 构建日频聚合缓存（{len(months)} 个月，flock 互斥）…",
                  flush=True)
            _build_with_lock()
        df = pd.read_parquet(CACHE_PATH)
        df['date'] = pd.to_datetime(df['date'])
        idx = pd.MultiIndex.from_arrays(
            [df['stock_code'].to_numpy(), df['date'].to_numpy()])
        _CACHE_SERIES = {c: pd.Series(df[c].to_numpy(dtype=float), index=idx)
                         for c in FACTOR_COLS}
        _CACHE_DF = df
        return _CACHE_DF
    except Exception as e:
        if not _WARNED:
            print(f"warning: intraday 缓存加载失败 {str(e)[:150]}，因子全 NaN",
                  flush=True)
            _WARNED = True
        return None


def _lookup(frame: pd.DataFrame, col: str) -> np.ndarray:
    """按 (stock_code, date) 对齐取因子值；缓存缺失 → 全 NaN。"""
    if _load_cache() is None:
        return np.full(len(frame), np.nan)
    key = pd.MultiIndex.from_arrays([
        frame['stock_code'].astype(str).to_numpy(),
        pd.to_datetime(frame['date']).to_numpy()])
    return _CACHE_SERIES[col].reindex(key).to_numpy(dtype=float)


@register_py_factor(
    factor_id='id_hi_amt_share20', axis='intraday',
    description='高位成交占比：收盘处于当日[low,high]前20%区间的5分钟成交额/全天（>50%且收上影=高位派发嫌疑）')
def id_hi_amt_share20(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'hi_amt_share20')


@register_py_factor(
    factor_id='id_hi_shadow_dump', axis='intraday',
    description='高位派发强度：高位成交占比 × 日上影比例（铁血出货交互项）')
def id_hi_shadow_dump(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'hi_shadow_dump')


@register_py_factor(
    factor_id='id_late30_ret', axis='intraday',
    description='尾盘30分钟动量：close(15:00)/close(14:30)-1（T+1抢筹资金）')
def id_late30_ret(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'late30_ret')


@register_py_factor(
    factor_id='id_late30_amt_share', axis='intraday',
    description='尾盘30分钟成交占比：amount(14:35~15:00)/全天amount')
def id_late30_amt_share(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'late30_amt_share')


@register_py_factor(
    factor_id='id_updown_vol_ratio', axis='intraday',
    description='日内上下行波动比：std(阳线5分钟收益)/std(阴线5分钟收益)')
def id_updown_vol_ratio(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'updown_vol_ratio')


@register_py_factor(
    factor_id='id_dd_recovery', axis='intraday',
    description='日内最大回撤修复度：(日终close-回撤谷底)/(回撤峰-谷底)；无有效回撤(<0.5%)=1.0；截断[0,1]')
def id_dd_recovery(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'dd_recovery')


# ---- 3日滚动均值平滑变体（降单日脉冲噪声与换手；原版6个保持裸值对照） ----

@register_py_factor(
    factor_id='id_hi_amt_share20_s3', axis='intraday',
    description='高位成交占比 3日均（降单日脉冲，换手更低）')
def id_hi_amt_share20_s3(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'hi_amt_share20_s3')


@register_py_factor(
    factor_id='id_hi_shadow_dump_s3', axis='intraday',
    description='高位派发强度 3日均（铁血出货稳健版）')
def id_hi_shadow_dump_s3(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'hi_shadow_dump_s3')


@register_py_factor(
    factor_id='id_updown_vol_ratio_s3', axis='intraday',
    description='日内上下行波动比 3日均（波动不对称稳健版）')
def id_updown_vol_ratio_s3(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'updown_vol_ratio_s3')


@register_py_factor(
    factor_id='id_dd_recovery_s3', axis='intraday',
    description='日内回撤修复度 3日均（修复行为稳健版）')
def id_dd_recovery_s3(frame: pd.DataFrame, ev) -> np.ndarray:
    return _lookup(frame, 'dd_recovery_s3')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='intraday 日频聚合缓存构建')
    ap.add_argument('--build', action='store_true', help='构建/增量检查缓存')
    ap.add_argument('--force', action='store_true', help='强制全量重建')
    ap.add_argument('--stats', action='store_true', help='打印缓存统计')
    a = ap.parse_args()
    if a.build or a.force:
        build_cache(force=a.force)
    if a.stats:
        df = _load_cache()
        if df is None:
            print('cache 不可用')
        else:
            print(f"rows={len(df):,} date={df['date'].min().date()}~{df['date'].max().date()}")
            for c in FACTOR_COLS:
                v = df[c]
                print(f"  {c}: finite={np.isfinite(v).mean():.3f} "
                      f"mean={v.mean():.5f} std={v.std():.5f} "
                      f"p1={v.quantile(0.01):.4f} p99={v.quantile(0.99):.4f}")
