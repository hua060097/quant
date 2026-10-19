"""分钟前复权树构建（经验对齐物化版，2026-09-08 用户定版）。

k(t,stock) = 日线open / 当日分钟首bar open；全天OHLC同乘；volume不动。
规则：
- 日线无行（停牌日）→ 整天丢弃（计halt_dropped + 样例日志备查）；
- |k-1| > --k_warn → 照转 + WARNING进日志（不fallback，后续按日志案例核查日线/分钟谁错）；
- |k-1| <= 1e-9 → 原样通过（无浮点尘）；
- 日线open缺失/非正、分钟首bar缺失/非正 → 丢弃+计数。
raw原始树永不动；输出独立目录；行数对账（raw == kept + dropped_*）。

用法：
    python build_minute_adj_tree.py --start_yyyymm 202306 --end_yyyymm 202609
        [--minute_root ../local_data/minute] [--out_root ../local_data/minute_adj]
        [--daily_cache auto] [--k_warn 0.5] [--dry_run]
--daily_cache auto = cache目录下"覆盖跨度≥28天"文件中止期最新者（单日增量文件
自动排除，防止抢源）；日线止期不足最晚待处理月时会 WARNING 提示覆盖缺口。
manifest.csv 按月份合并写入（增量转换不丢历史月份统计）。
"""
import argparse
import glob
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_K_TOL_IDENTITY = 1e-9


def _setup_logger(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"build_minute_adj_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("build_minute_adj")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(f"日志文件: {log_path}")
    return logger


def process_month(minute_file: Path, cache_file: Path, out_file: Path, k_warn: float,
                  logger, dry_run: bool = False) -> dict:
    yyyymm = minute_file.parent.name
    daily = pd.read_parquet(
        str(cache_file), columns=["stock_code", "date", "open"],
        filters=[[("date", ">=", datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1)),
                  ("date", "<", datetime(int(yyyymm[:4]) + (1 if yyyymm[4:6] == "12" else 0),
                                         int(yyyymm[4:6]) % 12 + 1, 1))]])
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y%m%d")
    daily["stock_code"] = daily["stock_code"].astype(str)
    daily["open"] = pd.to_numeric(daily["open"], errors="coerce")
    daily = daily.dropna(subset=["open"])
    daily = daily[daily["open"] > 0]
    daily_map = dict(zip(daily["stock_code"] + "|" + daily["date"], daily["open"].astype(float)))
    df = pd.read_parquet(str(minute_file))
    n_read = len(df)
    if df.empty:
        return {"month": minute_file.parent.name, "raw_rows": 0, "kept_rows": 0,
                "dropped_halt": 0, "dropped_coverage": 0, "dropped_zerovol": 0,
                "dropped_badtime": 0,
                "bad_daily": 0, "bad_minute": 0,
                "scaled_days": 0, "identity_days": 0, "extreme_days": 0,
                "extreme_list": ""}
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    n_badtime = int(df["datetime"].isna().sum())
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    n_raw = len(df)
    df["_s"] = df["stock_code"].astype(str)
    df["_d"] = df["datetime"].dt.strftime("%Y%m%d")
    df["_key"] = df["_s"] + "|" + df["_d"]
    firsts = df.drop_duplicates(["_s", "_d"], keep="first").set_index("_key")["open"]
    firsts = pd.to_numeric(firsts, errors="coerce")

    do_map = {k: daily_map.get(k) for k in firsts.index}
    # 强制 float dtype：全 None（该月完全无日线，如缓存未覆盖）时避免 object 比较崩溃
    do = pd.Series({k: (v if v is not None else np.nan) for k, v in do_map.items()},
                   dtype=float)
    mo = firsts.astype(float)
    has_daily = do.notna() & (do > 0)
    has_minute = mo.notna() & (mo > 0)
    valid = has_daily & has_minute
    k = pd.Series(np.nan, index=firsts.index, dtype=float)
    k[valid] = (do[valid].astype(float) / mo[valid].astype(float)).astype(float)

    # 日成交量(区分真停牌零量与覆盖缺失)
    dayvol = df.groupby("_key")["volume"].sum() if "volume" in df.columns \
        else pd.Series(1.0, index=firsts.index)
    dayvol = pd.to_numeric(dayvol, errors="coerce").fillna(0.0)

    # 互斥丢弃：缺日线(覆盖缺口) > 分钟全天零量(真停牌幻影) > 坏值
    missing_daily = set(firsts.index[~firsts.index.isin(do.dropna().index)])
    zero_vol = set(dayvol[dayvol <= 0].index) - missing_daily
    bad_daily = set(firsts.index[do.notna() & (do <= 0)]) - missing_daily
    bad_minute = set(firsts.index[has_daily & ~has_minute]) - missing_daily - zero_vol
    halt_keys = missing_daily | zero_vol | bad_daily | bad_minute
    bad_daily = len(bad_daily)
    bad_minute = len(bad_minute)

    keep_idx = firsts.index[valid & ~firsts.index.isin(zero_vol)]
    k_kept = k.loc[keep_idx]
    identity = keep_idx[np.abs(k_kept - 1.0) <= _K_TOL_IDENTITY]
    scaled = keep_idx[np.abs(k_kept - 1.0) > _K_TOL_IDENTITY]
    extreme = keep_idx[np.abs(k_kept - 1.0) > k_warn]
    # 尖峰vs漂移：相对该股本月内trailing中位k偏离>20%才告警(真异常)，
    # 持续偏离(锚漂移)只计数不刷屏
    _kframe = pd.DataFrame({"key": k_kept.index, "k": k_kept.values})
    _kframe["_s"] = _kframe["key"].astype(str).str.split("|").str[0]
    _kframe["_d"] = _kframe["key"].astype(str).str.split("|").str[1]
    _kframe = _kframe.sort_values(["_s", "_d"])
    _kframe["_trail"] = _kframe.groupby("_s")["k"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=5).median())
    _spike = _kframe.set_index("key")["_trail"]
    # trail NaN(每月前5天无足够历史)→归入drift，不判spike，避免月初误报
    _spike = extreme[_spike.loc[extreme].notna()
                     & (np.abs(k_kept.loc[extreme] - _spike.loc[extreme]) > 0.20)]
    ext_list = [(str(kk).split("|")[0], str(kk).split("|")[1], round(float(k_kept[kk]), 6))
                for kk in extreme]
    for _s, _d, _k in ext_list[:50]:
        _tag = "SPIKE" if f"{_s}|{_d}" in set(_spike.astype(str)) else "drift"
        logger.warning(f"   [k-{_tag}] {_s} {_d} k={_k}（照转+记录，超{k_warn}阈值，待核查日线/分钟谁错）")
    if len(ext_list) > 50:
        logger.warning(f"   [k-extreme] ...共{len(ext_list)}天超阈值(仅打印前50，详见manifest)")

    keep_keys = set(keep_idx)
    out = df[df["_key"].isin(keep_keys)].copy()
    kmap = k_kept.to_dict()
    out["_k"] = out["_key"].map(kmap)
    for _c in ("open", "high", "low", "close"):
        if _c in out.columns:
            out[_c] = pd.to_numeric(out[_c], errors="coerce") * out["_k"]
    out = out.drop(columns=["_s", "_d", "_key", "_k"])

    halt_sample = sorted(missing_daily)[:5] + sorted(zero_vol)[:5]
    if halt_sample:
        logger.info(f"   丢弃 {len(halt_keys)} 天(覆盖缺口{len(missing_daily)}+真停牌零量{len(zero_vol)}+坏值{bad_daily + bad_minute}), 样例: {halt_sample}")

    stat = {"month": minute_file.parent.name, "raw_rows": int(n_read),
            "kept_rows": int(len(out)), "dropped_halt": int(len(halt_keys)),
            "dropped_coverage": int(len(missing_daily)), "dropped_zerovol": int(len(zero_vol)),
            "dropped_badtime": int(n_badtime),
            "bad_daily": int(bad_daily), "bad_minute": int(bad_minute),
            "scaled_days": int(len(scaled)), "identity_days": int(len(identity)),
            "extreme_days": int(len(extreme)),
            "extreme_list": ";".join(f"{s}/{d}/{k}" for s, d, k in ext_list)}
    assert int(len(out)) + int(len(df[~df["_key"].isin(keep_keys)])) + int(n_badtime) \
        == int(n_read), "行数对账失败"
    if not dry_run:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(str(out_file), index=False)
    return stat


def main() -> int:
    p = argparse.ArgumentParser(description="分钟前复权树构建（经验对齐物化版）")
    p.add_argument("--start_yyyymm", default=None)
    p.add_argument("--end_yyyymm", default=None)
    p.add_argument("--minute_root", default="../local_data/minute")
    p.add_argument("--out_root", default="../local_data/minute_adj")
    p.add_argument("--daily_cache", default="auto")
    p.add_argument("--k_warn", type=float, default=0.5)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    minute_root = (Path.cwd() / args.minute_root).resolve() if not os.path.isabs(args.minute_root) \
        else Path(args.minute_root)
    out_root = (Path.cwd() / args.out_root).resolve() if not os.path.isabs(args.out_root) \
        else Path(args.out_root)
    if out_root == minute_root or str(out_root).startswith(str(minute_root) + os.sep):
        print("拒绝：out_root 不能是 minute_root 本身或其子目录（保护原始数据）")
        return 2
    logger = _setup_logger(out_root if not args.dry_run
                           else Path(tempfile.gettempdir()))
    logger.info(f"minute_root={minute_root} out_root={out_root} dry_run={args.dry_run}")

    months = sorted(d.name for d in minute_root.iterdir()
                    if d.is_dir() and d.name.isdigit() and len(d.name) == 6)
    if args.start_yyyymm:
        months = [m for m in months if m >= args.start_yyyymm]
    if args.end_yyyymm:
        months = [m for m in months if m <= args.end_yyyymm]

    cache_dir = _HERE / "cache"
    if args.daily_cache == "auto":
        import re as _re
        cands = []
        for _p in glob.glob(str(cache_dir / "stock_features_*.parquet")):
            _m = _re.search(r"_(\d{8})_(\d{8})\.parquet$", Path(_p).name)
            if not _m:
                continue
            _s, _e = _m.group(1), _m.group(2)
            _span = (datetime.strptime(_e, "%Y%m%d")
                     - datetime.strptime(_s, "%Y%m%d")).days
            # 排除单日增量文件：止期最新 ≠ 覆盖最全（每日增量会抢走 auto 选择）
            if _span >= 28:
                cands.append((_e, _span, Path(_p)))
        if not cands:
            logger.error("cache 下无覆盖≥28天的 stock_features_*.parquet"
                         "（单日增量文件不能作为日线源，请用 --daily_cache 显式指定）")
            return 2
        _e, _span, cache_file = sorted(cands)[-1]
        if months:
            _last = months[-1]
            _need = (datetime(int(_last[:4]) + (1 if _last[4:6] == "12" else 0),
                              int(_last[4:6]) % 12 + 1, 1)
                     - pd.Timedelta(days=1))
            _have = datetime.strptime(_e, "%Y%m%d")
            if _have < _need:
                logger.warning(f"日线源止期 {_have.date()} < 待处理最晚月 {_last} 月末 "
                               f"{_need.date()}：超出部分将按覆盖缺口整天丢弃")
    else:
        cache_file = Path(args.daily_cache)
    logger.info(f"日线源: {cache_file.name}")
    logger.info(f"待处理月份: {months}")
    stats = []
    for m in months:
        mf = minute_root / m / "minute_data.parquet"
        if not mf.exists():
            logger.warning(f"跳过缺失月份文件: {mf}")
            continue
        logger.info(f"== 处理 {m} ==")
        st = process_month(mf, cache_file, out_root / m / "minute_data.parquet",
                           args.k_warn, logger, args.dry_run)
        stats.append(st)
    man = pd.DataFrame(stats)
    if not args.dry_run:
        man_path = out_root / "manifest.csv"
        man["month"] = man["month"].astype(str)
        if man_path.exists() and len(man):
            try:
                _old = pd.read_csv(man_path, encoding="utf-8-sig")
                _old["month"] = _old["month"].astype(str)
                _old = _old[~_old["month"].isin(man["month"])]
                man = pd.concat([_old, man], ignore_index=True).sort_values("month")
            except Exception as _e:
                logger.warning(f"旧 manifest 合并失败（{str(_e)[:100]}），仅写入本次统计")
        man.to_csv(man_path, index=False, encoding="utf-8-sig")
        logger.info("manifest已保存（按月份合并，历史月份保留）")
    else:
        logger.info("[dry_run] 不写盘")
        print(man.to_string(index=False))
    tot = man.select_dtypes(include=[np.number]).sum(numeric_only=True)
    logger.info(f"合计: raw={int(tot.get('raw_rows', 0)):,} kept={int(tot.get('kept_rows', 0)):,} "
                f"halt_drop={int(tot.get('dropped_halt', 0)):,} scaled={int(tot.get('scaled_days', 0)):,} "
                f"extreme={int(tot.get('extreme_days', 0)):,} identity={int(tot.get('identity_days', 0)):,} "
                f"(天数口径)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
