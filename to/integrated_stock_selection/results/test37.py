import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# 基础统计
# =========================
def calc_max_drawdown(rets: pd.Series) -> float:
    if len(rets) == 0:
        return np.nan
    nav = (1 + rets.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def calc_sharpe(rets: pd.Series) -> float:
    if len(rets) == 0:
        return np.nan
    std = rets.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(rets.mean() / std * math.sqrt(252))


def calc_cvar10(rets: pd.Series) -> float:
    if len(rets) == 0:
        return np.nan
    q = rets.quantile(0.10)
    tail = rets[rets <= q]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def summarize_returns(name: str, rets: List[float]) -> Dict:
    s = pd.Series(rets, dtype=float).dropna()
    if len(s) == 0:
        return {
            "strategy": name,
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "win_rate": np.nan,
            "big_loss_3pct_rate": np.nan,
            "big_loss_5pct_rate": np.nan,
            "big_loss_8pct_rate": np.nan,
            "cvar10": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "final_nav": np.nan,
        }

    nav = float((1 + s).prod())
    return {
        "strategy": name,
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "win_rate": float((s > 0).mean()),
        "big_loss_3pct_rate": float((s <= -0.03).mean()),
        "big_loss_5pct_rate": float((s <= -0.05).mean()),
        "big_loss_8pct_rate": float((s <= -0.08).mean()),
        "cvar10": calc_cvar10(s),
        "max_drawdown": calc_max_drawdown(s),
        "sharpe": calc_sharpe(s),
        "final_nav": nav,
    }


def print_summary(title: str, rets: List[float]) -> None:
    stat = summarize_returns(title, rets)
    print("=" * 120)
    print(title)
    print("=" * 120)
    for k, v in stat.items():
        if k == "strategy":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print()


# =========================
# 数据读取与工具
# =========================
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 按日期排序
    return dict(sorted(data.items(), key=lambda x: x[0]))


def safe_get_topn(day_obj: Dict, key: str) -> List[Dict]:
    arr = day_obj.get(key, [])
    if not isinstance(arr, list):
        return []
    return arr


def dedup_keep_best_by_code(rows: List[Dict], score_key: str) -> List[Dict]:
    best = {}
    for r in rows:
        code = r.get("code")
        if code is None:
            continue
        cur = best.get(code)
        score = r.get(score_key, np.nan)
        if cur is None:
            best[code] = r
        else:
            cur_score = cur.get(score_key, np.nan)
            if pd.isna(cur_score) or (not pd.isna(score) and score > cur_score):
                best[code] = r
    out = list(best.values())
    out.sort(key=lambda x: x.get(score_key, -1e18), reverse=True)
    return out


def zscore_cross_section(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd == 0 or np.isnan(sd):
        return [0.0] * len(arr)
    return ((arr - mu) / sd).tolist()


def pick_top1_by_rank(rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None
    rows2 = sorted(rows, key=lambda x: x.get("rank_score", -1e18), reverse=True)
    return rows2[0]


def pick_top1_by_cls(rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None
    rows2 = sorted(rows, key=lambda x: x.get("p_cls", -1e18), reverse=True)
    return rows2[0]


def code_to_rank(rows: List[Dict]) -> Dict[str, int]:
    d = {}
    for i, r in enumerate(rows, start=1):
        code = r.get("code")
        if code is not None and code not in d:
            d[code] = i
    return d


# =========================
# 单日策略构造
# =========================
def strategy_rank_top1(day_obj: Dict) -> Optional[Dict]:
    return day_obj.get("rank_all_top1")


def strategy_serial_top1(day_obj: Dict) -> Optional[Dict]:
    return day_obj.get("serial_top1")


def strategy_rank_top20_cls_top5_rank1(day_obj: Dict, top_rank_n: int = 20, top_cls_n: int = 5) -> Optional[Dict]:
    rank_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "rank_topn"), "rank_score")
    if len(rank_topn) == 0:
        return None

    rank_topn = rank_topn[:top_rank_n]
    rank_topn = sorted(rank_topn, key=lambda x: x.get("p_cls", -1e18), reverse=True)[:top_cls_n]
    return pick_top1_by_rank(rank_topn)


def strategy_cls_top20_rank_top5_cls1(day_obj: Dict, top_cls_n: int = 20, top_rank_n: int = 5) -> Optional[Dict]:
    cls_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "cls_topn"), "p_cls")
    if len(cls_topn) == 0:
        return None

    cls_topn = cls_topn[:top_cls_n]
    cls_topn = sorted(cls_topn, key=lambda x: x.get("rank_score", -1e18), reverse=True)[:top_rank_n]
    return pick_top1_by_cls(cls_topn)


def strategy_intersection_rank(day_obj: Dict, n_rank: int = 20, n_cls: int = 20) -> Optional[Dict]:
    rank_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "rank_topn"), "rank_score")[:n_rank]
    cls_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "cls_topn"), "p_cls")[:n_cls]

    cls_codes = {x["code"] for x in cls_topn if "code" in x}
    inter = [x for x in rank_topn if x.get("code") in cls_codes]
    return pick_top1_by_rank(inter)


def strategy_intersection_cls(day_obj: Dict, n_rank: int = 20, n_cls: int = 20) -> Optional[Dict]:
    rank_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "rank_topn"), "rank_score")[:n_rank]
    cls_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "cls_topn"), "p_cls")[:n_cls]

    rank_codes = {x["code"] for x in rank_topn if "code" in x}
    inter = [x for x in cls_topn if x.get("code") in rank_codes]
    return pick_top1_by_cls(inter)


def strategy_fused(day_obj: Dict, top_cls_n: int = 20, lam: float = 0.3) -> Optional[Dict]:
    cls_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "cls_topn"), "p_cls")[:top_cls_n]
    if len(cls_topn) == 0:
        return None

    pcls_list = [x.get("p_cls", np.nan) for x in cls_topn]
    zcls_list = zscore_cross_section(pcls_list)

    rows = []
    for r, zc in zip(cls_topn, zcls_list):
        rr = dict(r)
        rr["_fused"] = rr.get("rank_score", -1e18) + lam * zc
        rows.append(rr)

    rows.sort(key=lambda x: x.get("_fused", -1e18), reverse=True)
    return rows[0] if rows else None


# =========================
# overlap / 位置统计
# =========================
def analyze_overlap(data: Dict, n_rank: int = 20, n_cls: int = 20) -> pd.DataFrame:
    rows = []

    for dt, day_obj in data.items():
        rank_top1 = day_obj.get("rank_all_top1", {})
        serial_top1 = day_obj.get("serial_top1", {})

        rank_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "rank_topn"), "rank_score")[:n_rank]
        cls_topn = dedup_keep_best_by_code(safe_get_topn(day_obj, "cls_topn"), "p_cls")[:n_cls]

        rank_rank = code_to_rank(rank_topn)
        cls_rank = code_to_rank(cls_topn)

        rank_top1_code = rank_top1.get("code")
        serial_top1_code = serial_top1.get("code")

        rows.append({
            "date": dt,
            "rank_top1_code": rank_top1_code,
            "serial_top1_code": serial_top1_code,
            "rank_top1_in_cls_top20": int(rank_top1_code in cls_rank) if rank_top1_code else 0,
            "rank_top1_cls_rank": cls_rank.get(rank_top1_code, np.nan),
            "serial_top1_in_rank_top20": int(serial_top1_code in rank_rank) if serial_top1_code else 0,
            "serial_top1_rank_rank": rank_rank.get(serial_top1_code, np.nan),
            "same_pick": int(rank_top1_code == serial_top1_code) if rank_top1_code and serial_top1_code else 0,
            "rank_top1_ret": rank_top1.get("real_return", np.nan),
            "serial_top1_ret": serial_top1.get("real_return", np.nan),
        })

    return pd.DataFrame(rows)


def print_overlap_report(df: pd.DataFrame) -> None:
    print("=" * 120)
    print("OVERLAP / RANK DISTRIBUTION")
    print("=" * 120)

    print(f"days: {len(df)}")
    print(f"same_pick_rate: {df['same_pick'].mean():.6f}")
    print(f"rank_top1_in_cls_top20_rate: {df['rank_top1_in_cls_top20'].mean():.6f}")
    print(f"serial_top1_in_rank_top20_rate: {df['serial_top1_in_rank_top20'].mean():.6f}")
    print()

    print("rank_top1_cls_rank distribution:")
    print(df["rank_top1_cls_rank"].value_counts(dropna=False).sort_index())
    print()

    print("serial_top1_rank_rank distribution:")
    print(df["serial_top1_rank_rank"].value_counts(dropna=False).sort_index())
    print()

    not_in_cls = df[df["rank_top1_in_cls_top20"] == 0]
    in_cls = df[df["rank_top1_in_cls_top20"] == 1]
    if len(not_in_cls) > 0:
        print(f"rank_top1 NOT in cls_top20: {len(not_in_cls)} days, mean_ret={not_in_cls['rank_top1_ret'].mean():.6f}")
    if len(in_cls) > 0:
        print(f"rank_top1 IN cls_top20: {len(in_cls)} days, mean_ret={in_cls['rank_top1_ret'].mean():.6f}")
    print()


# =========================
# 批量回测
# =========================
def run_strategy(data: Dict, fn, name: str) -> Tuple[List[float], pd.DataFrame]:
    rets = []
    picks = []

    for dt, day_obj in data.items():
        pick = fn(day_obj)
        if pick is None:
            continue
        ret = pick.get("real_return", np.nan)
        if pd.isna(ret):
            continue

        rets.append(float(ret))
        picks.append({
            "date": dt,
            "code": pick.get("code"),
            "real_return": ret,
            "rank_score": pick.get("rank_score", pick.get("predicted_return", np.nan)),
            "p_cls": pick.get("p_cls", np.nan),
            "rank_pos_all": pick.get("rank_pos_all", np.nan),
            "rank_pct_all": pick.get("rank_pct_all", np.nan),
        })

    return rets, pd.DataFrame(picks)


def compare_strategies(data: Dict) -> pd.DataFrame:
    strategy_defs = [
        ("rank_top1", lambda d: strategy_rank_top1(d)),
        ("serial_top1", lambda d: strategy_serial_top1(d)),
        ("rank_top20_cls_top5_rank1", lambda d: strategy_rank_top20_cls_top5_rank1(d, 20, 5)),
        ("cls_top20_rank_top5_cls1", lambda d: strategy_cls_top20_rank_top5_cls1(d, 20, 5)),
        ("intersection_rank", lambda d: strategy_intersection_rank(d, 20, 20)),
        ("intersection_cls", lambda d: strategy_intersection_cls(d, 20, 20)),
        ("fused_lam_0.2", lambda d: strategy_fused(d, 20, 0.2)),
        ("fused_lam_0.3", lambda d: strategy_fused(d, 20, 0.3)),
        ("fused_lam_0.5", lambda d: strategy_fused(d, 20, 0.5)),
    ]

    res = []
    for name, fn in strategy_defs:
        rets, _ = run_strategy(data, fn, name)
        res.append(summarize_returns(name, rets))
    return pd.DataFrame(res).sort_values(["mean", "sharpe"], ascending=[False, False])


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="daily_gate_stats json path")
    args = parser.parse_args()

    data = load_json(args.json)

    print(f"Loaded days: {len(data)}")
    first_day = next(iter(data.keys()))
    print(f"First day: {first_day}")
    print()

    # baseline
    baseline_rets, _ = run_strategy(data, lambda d: strategy_rank_top1(d), "rank_top1")
    serial_rets, _ = run_strategy(data, lambda d: strategy_serial_top1(d), "serial_top1")

    print_summary("BASELINE | rank_top1", baseline_rets)
    print_summary("BASELINE | serial_top1(cls_topN -> rank_top1)", serial_rets)

    # overlap
    overlap_df = analyze_overlap(data, 20, 20)
    print_overlap_report(overlap_df)

    # strategies
    comp = compare_strategies(data)
    print("=" * 120)
    print("STRATEGY COMPARISON")
    print("=" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(comp.to_string(index=False))
    print()

    # 额外导出
    out_csv = Path(args.json).with_suffix("").as_posix() + "_strategy_compare.csv"
    comp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved strategy comparison to: {out_csv}")

    overlap_csv = Path(args.json).with_suffix("").as_posix() + "_overlap_report.csv"
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    print(f"Saved overlap report to: {overlap_csv}")


if __name__ == "__main__":
    main()