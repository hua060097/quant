"""
分析四个涨停表现因子是否能作为 three_rule_v3 新规则：
  prev_first_limit_up_performance / prev_first_limit_up_performance_percentile_20d
  prev_non_first_limit_up_performance / prev_non_first_limit_up_performance_percentile_20d

核心问题：能否拦截大亏，且不误杀大涨？
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"

# 最新三个运行文件 (与 analyze_recent_loss_factor_separation.py 一致)
RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}

NEW_FACTORS = [
    "prev_first_limit_up_performance",
    "prev_first_limit_up_performance_percentile_20d",
    "prev_non_first_limit_up_performance",
    "prev_non_first_limit_up_performance_percentile_20d",
]

# 现有 three_rule_v3 规则 (用于对比)
EXISTING_V3_MASK = """
((runup_ma5_exhaustion > 0.8) & (cpv_factor <= -0.4))
| (((prev_day_blowoff_risk >= 2.0) & (sector_momentum_accel_rank <= 0.30))
   | ((prev_day_blowoff_risk > 5.0) & (sector_momentum_accel_rank <= 0.50))
   | ((prev_day_blowoff_risk > 1.0) & (sector_momentum_accel_rank <= 0.15)))
| ((sector_leadership_persistence_v3 <= -0.18) & (return_20d_intra_z <= 0.40))
| ((market_liquidity_change < 0.94) & ((ulcer_index < -0.8) | (momentum_peak_decay_3d > 1.0) | (intra_sector_momentum_rank < 0.67)))
| ((mkt_dispersion_shock_v2 < 0.98) & (market_advance_decline_ratio < 0.38))
| (limit_down_3d == 1)
| ((top10_10d_strength_level_v3 >= 0.70) & (buy_sm_amount_rate <= -1.25))
"""


def load_events():
    rows = []
    for ds, path in RUN_FILES.items():
        if not path.exists():
            print(f"[WARN] {path} not found, skip")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        key = list(data[0]["all_results"].keys())[0]
        for t in data[0]["all_results"][key]["trades"]:
            rows.append({
                "dataset": ds,
                "stock_code": t["stock_code"],
                "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                "sell_date": pd.Timestamp(t["sell_datetime"][:10]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
            })
    events = pd.DataFrame(rows)
    events["key"] = events["stock_code"] + "|" + events["buy_date"].dt.strftime("%Y-%m-%d")
    return events


def enrich_with_new_factors(events):
    """用 factors.parquet 中的新因子丰富交易事件"""
    df_all = pd.read_parquet(FACTOR_FILE)
    factor_cols = [c for c in NEW_FACTORS if c in df_all.columns]
    missing = [c for c in NEW_FACTORS if c not in df_all.columns]
    if missing:
        print(f"[WARN] 以下因子不在 parquet 中: {missing}")

    cols = ["date", "stock_code"] + factor_cols
    fac = df_all[cols].copy()
    del df_all
    fac["date"] = pd.to_datetime(fac["date"])
    fac = fac.sort_values(["stock_code", "date"])

    # 按股票分组建立索引
    grouped = {}
    for stock, g in fac.groupby("stock_code", sort=False):
        grouped[stock] = (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))

    rows = []
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is None:
            rows.append(rec)
            continue
        dates, g = item
        pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left") - 1
        if pos >= 0:
            r = g.iloc[int(pos)]
            rec["factor_date"] = r["date"]
            for c in factor_cols:
                if c in r.index:
                    rec[c] = r[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def current_v3_mask(df):
    return (
        ((df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4))
        | (
            ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
            | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
            | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
        )
        | ((df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40))
        | (
            (df["market_liquidity_change"] < 0.94)
            & (
                (df["ulcer_index"] < -0.8)
                | (df["momentum_peak_decay_3d"] > 1.0)
                | (df["intra_sector_momentum_rank"] < 0.67)
            )
        )
        | ((df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38))
        | (df.get("limit_down_3d", pd.Series(0, index=df.index)).fillna(0).astype(float) == 1)
        | (
            (df.get("top10_10d_strength_level_v3", pd.Series(np.nan, index=df.index)) >= 0.70)
            & (df.get("buy_sm_amount_rate", pd.Series(np.nan, index=df.index)) <= -1.25)
        )
    ).fillna(False)


def dedup_reference(df):
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    x = df.copy()
    x["priority"] = x["dataset"].map(priority)
    x = x.sort_values(["key", "priority"])
    return x.drop_duplicates("key", keep="first").drop(columns=["priority"])


def zdist(a, b):
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    av = av[np.isfinite(av)]
    bv = bv[np.isfinite(bv)]
    if len(av) < 2 or len(bv) < 3:
        return np.nan
    pooled = np.nanstd(np.r_[av, bv])
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (np.nanmedian(av) - np.nanmedian(bv)) / pooled


def main():
    print("=" * 80)
    print("分析四个涨停表现因子对 three_rule_v3 的补充价值")
    print("=" * 80)

    # 1. 加载并去重交易事件
    events = load_events()
    print(f"\n总交易事件: {len(events)} (datasets: {events['dataset'].value_counts().to_dict()})")

    # 2. 富化新因子
    df = enrich_with_new_factors(events)
    print(f"富化后行数: {len(df)}")

    # 检查新因子覆盖率
    for c in NEW_FACTORS:
        if c in df.columns:
            non_null = df[c].notna().sum()
            print(f"  {c}: {non_null}/{len(df)} non-null ({non_null/len(df)*100:.1f}%)")
        else:
            print(f"  {c}: MISSING")

    # 3. 标记现有 v3 规则命中
    # 先需要富化 v3 所需因子
    df_all_factors = pd.read_parquet(FACTOR_FILE)
    v3_cols = [
        "runup_ma5_exhaustion", "cpv_factor", "prev_day_blowoff_risk",
        "sector_momentum_accel_rank", "sector_leadership_persistence_v3",
        "return_20d_intra_z", "market_liquidity_change", "ulcer_index",
        "momentum_peak_decay_3d", "intra_sector_momentum_rank",
        "mkt_dispersion_shock_v2", "market_advance_decline_ratio",
        "limit_down_3d", "top10_10d_strength_level_v3", "buy_sm_amount_rate",
    ]
    needed_cols = ["date", "stock_code"] + [c for c in v3_cols if c in df_all_factors.columns]
    fac = df_all_factors[needed_cols].copy()
    del df_all_factors
    fac["date"] = pd.to_datetime(fac["date"])
    fac = fac.sort_values(["stock_code", "date"])
    grouped = {}
    for stock, g in fac.groupby("stock_code", sort=False):
        grouped[stock] = (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
    for c in v3_cols:
        if c not in df.columns:
            df[c] = np.nan
    for idx, row in df.iterrows():
        item = grouped.get(row["stock_code"])
        if item is None:
            continue
        dates, g = item
        pos = np.searchsorted(dates, np.datetime64(row["buy_date"]), side="left") - 1
        if pos >= 0:
            r = g.iloc[int(pos)]
            for c in v3_cols:
                if c in r.index and c in df.columns:
                    df.at[idx, c] = r[c]

    df["current_v3"] = current_v3_mask(df)

    # 4. 定义关键子集
    ref = dedup_reference(df)
    big_loss = ref[ref["actual_return"] <= -0.05].copy()
    big_win = ref[ref["actual_return"] >= 0.10].copy()
    # 现有 v3 漏网大亏 (recent)
    recent_leak = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
        & ~df["current_v3"]
    ].copy()
    # 现有 v3 命中的大亏
    v3_caught_loss = df[
        (df["actual_return"] <= -0.05)
        & df["current_v3"]
    ].copy()
    # 大涨且被 v3 误杀
    v3_false_kill = df[
        (df["actual_return"] >= 0.10)
        & df["current_v3"]
    ].copy()

    print(f"\n== 样本统计 ==")
    print(f"大亏 (return <= -5%): {len(big_loss)}")
    print(f"大涨 (return >= 10%): {len(big_win)}")
    print(f"v3 漏网大亏 (recent 184, 4月后): {len(recent_leak)}")
    print(f"v3 拦截大亏: {len(v3_caught_loss)}")
    print(f"v3 误杀大涨: {len(v3_false_kill)}")

    # 5. 各因子在大亏/大涨中的分布
    print(f"\n{'='*80}")
    print("一、因子分布对比：大亏 vs 大涨 vs 全体")
    print(f"{'='*80}")
    for c in NEW_FACTORS:
        if c not in df.columns:
            continue
        all_vals = ref[c].dropna()
        loss_vals = big_loss[c].dropna()
        win_vals = big_win[c].dropna()
        leak_vals = recent_leak[c].dropna()

        print(f"\n--- {c} ---")
        if len(all_vals) == 0:
            print("  无数据")
            continue
        print(f"  全体:   n={len(all_vals):4d}, mean={all_vals.mean():.4f}, med={all_vals.median():.4f}, "
              f"P10={all_vals.quantile(0.10):.4f}, P90={all_vals.quantile(0.90):.4f}")
        print(f"  大亏:   n={len(loss_vals):4d}, mean={loss_vals.mean():.4f}, med={loss_vals.median():.4f}, "
              f"P10={loss_vals.quantile(0.10):.4f}, P90={loss_vals.quantile(0.90):.4f}")
        print(f"  大涨:   n={len(win_vals):4d}, mean={win_vals.mean():.4f}, med={win_vals.median():.4f}, "
              f"P10={win_vals.quantile(0.10):.4f}, P90={win_vals.quantile(0.90):.4f}")
        if len(leak_vals) >= 2:
            print(f"  漏网大亏: n={len(leak_vals):4d}, mean={leak_vals.mean():.4f}, med={leak_vals.median():.4f}, "
                  f"P10={leak_vals.quantile(0.10):.4f}, P90={leak_vals.quantile(0.90):.4f}")

        # Z-distance
        zw = zdist(loss_vals, win_vals)
        print(f"  z-dist(大亏 vs 大涨): {zw:.3f}")
        if len(leak_vals) >= 2:
            zw_leak = zdist(leak_vals, win_vals)
            print(f"  z-dist(漏网大亏 vs 大涨): {zw_leak:.3f}")

    # 6. 阈值扫描：只看"低值拦截"（涨停表现差 → 市场冰点）
    print(f"\n{'='*80}")
    print("二、阈值扫描：寻找能拦截大亏但不误杀大涨的阈值")
    print(f"  思路：这些因子是市场环境变量，低值=首板/连板次日表现差=市场情绪冰点")
    print(f"  规则形式：factor <= threshold → 拦截")
    print(f"{'='*80}")

    for c in NEW_FACTORS:
        if c not in df.columns:
            continue
        print(f"\n--- {c} ---")
        vals = ref[c].dropna()
        if len(vals) < 10:
            print("  数据不足")
            continue

        # 扫描多个阈值
        quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        print(f"  {'阈值':>10s} | {'大亏命中':>8s} | {'大涨误杀':>8s} | {'全体命中':>8s} | "
              f"{'漏网大亏命中':>12s} | {'拦截率':>6s} | {'误杀率':>6s} | {'F1':>6s}")
        print(f"  {'-'*85}")

        for q in quantiles:
            th = vals.quantile(q)
            loss_hit = (big_loss[c] <= th).sum()
            win_hit = (big_win[c] <= th).sum()
            all_hit = (ref[c] <= th).sum()
            leak_hit = (recent_leak[c] <= th).sum() if len(recent_leak) > 0 else 0

            loss_rate = loss_hit / max(len(big_loss), 1)
            win_rate = win_hit / max(len(big_win), 1)
            # F1: 平衡拦截率和低误杀率
            if loss_rate > 0 and win_rate < 1:
                precision = loss_hit / max(loss_hit + win_hit, 1)
                f1 = 2 * precision * loss_rate / max(precision + loss_rate, 1e-8)
            else:
                f1 = 0

            print(f"  {th:>10.4f} | {loss_hit:>4d}/{len(big_loss):<3d} | {win_hit:>4d}/{len(big_win):<3d} | "
                  f"{all_hit:>4d}/{len(ref):<4d} | {leak_hit:>4d}/{len(recent_leak):<7d} | "
                  f"{loss_rate:>5.1%} | {win_rate:>5.1%} | {f1:>5.3f}")

    # 7. 与现有 v3 规则的互补性分析
    print(f"\n{'='*80}")
    print("三、与现有 v3 规则的互补性")
    print(f"  问题：新因子能拦截的、是否已经是 v3 能拦截的？")
    print(f"{'='*80}")

    for c in NEW_FACTORS:
        if c not in df.columns:
            continue
        print(f"\n--- {c} ---")
        vals = ref[c].dropna()
        if len(vals) < 10:
            continue

        # 选一个有意义的阈值（P20 或中位数附近的低值）
        for q in [0.15, 0.20, 0.25]:
            th = vals.quantile(q)
            # 大亏中新因子命中但 v3 未命中
            loss_new_hit = big_loss[big_loss[c] <= th]
            loss_new_only = loss_new_hit[~loss_new_hit["current_v3"]]
            # 大涨中新因子命中
            win_new_hit = big_win[big_win[c] <= th]

            print(f"  阈值={th:.4f} (P{q*100:.0f}):")
            print(f"    大亏命中: {len(loss_new_hit)}, 其中 v3 已拦截: {len(loss_new_hit) - len(loss_new_only)}, "
                  f"新增拦截: {len(loss_new_only)}")
            print(f"    大涨误杀: {len(win_new_hit)}")
            if len(loss_new_only) > 0:
                print(f"    新增大亏拦截详情:")
                for _, r in loss_new_only.sort_values("actual_return").iterrows():
                    print(f"      {r['stock_code']} {r['buy_date'].strftime('%Y-%m-%d')} "
                          f"ret={r['actual_return']:.2%} {c}={r[c]:.4f}")

    # 8. 详细的大亏交易快照
    print(f"\n{'='*80}")
    print("四、大亏交易的新因子快照")
    print(f"{'='*80}")
    snapshot_cols = ["stock_code", "buy_date", "actual_return", "profit_loss", "current_v3"] + NEW_FACTORS
    snapshot_cols = [c for c in snapshot_cols if c in big_loss.columns]
    print(big_loss.sort_values("actual_return")[snapshot_cols].to_string(index=False))

    # 9. 漏网大亏的新因子快照
    if len(recent_leak) > 0:
        print(f"\n{'='*80}")
        print("五、v3 漏网大亏的新因子快照 (184_v3, 2026-04后)")
        print(f"{'='*80}")
        snapshot_cols = ["stock_code", "buy_date", "actual_return", "profit_loss"] + NEW_FACTORS
        snapshot_cols = [c for c in snapshot_cols if c in recent_leak.columns]
        print(recent_leak.sort_values("actual_return")[snapshot_cols].to_string(index=False))

    # 10. 因子与 v3 现有规则的重叠分析
    print(f"\n{'='*80}")
    print("六、新因子与现有 v3 各规则的重叠分析")
    print(f"  检查新因子命中的大亏是否与现有 7 条规则有共同模式")
    print(f"{'='*80}")

    # 逐规则分析
    rules = {
        "R1_runup_cpv": (ref["runup_ma5_exhaustion"] > 0.8) & (ref["cpv_factor"] <= -0.4),
        "R2_blowoff_sector": (
            ((ref["prev_day_blowoff_risk"] >= 2.0) & (ref["sector_momentum_accel_rank"] <= 0.30))
            | ((ref["prev_day_blowoff_risk"] > 5.0) & (ref["sector_momentum_accel_rank"] <= 0.50))
            | ((ref["prev_day_blowoff_risk"] > 1.0) & (ref["sector_momentum_accel_rank"] <= 0.15))
        ),
        "R3_sector_lead_ret20": (ref["sector_leadership_persistence_v3"] <= -0.18) & (ref["return_20d_intra_z"] <= 0.40),
        "R4_liq_ulcer": (ref["market_liquidity_change"] < 0.94) & (
            (ref["ulcer_index"] < -0.8) | (ref["momentum_peak_decay_3d"] > 1.0) | (ref["intra_sector_momentum_rank"] < 0.67)
        ),
        "R5_disp_adr": (ref["mkt_dispersion_shock_v2"] < 0.98) & (ref["market_advance_decline_ratio"] < 0.38),
        "R6_limit_down": ref.get("limit_down_3d", pd.Series(0, index=ref.index)).fillna(0).astype(float) == 1,
        "R7_top10_buysm": (ref.get("top10_10d_strength_level_v3", pd.Series(np.nan, index=ref.index)) >= 0.70) & (
            ref.get("buy_sm_amount_rate", pd.Series(np.nan, index=ref.index)) <= -1.25
        ),
    }

    for c in NEW_FACTORS:
        if c not in ref.columns:
            continue
        vals = ref[c].dropna()
        if len(vals) < 10:
            continue

        # 选择 P20 作为阈值
        th = vals.quantile(0.20)
        new_mask = ref[c] <= th

        print(f"\n--- {c} <= {th:.4f} (P20) ---")
        loss_new = big_loss[big_loss[c] <= th]
        win_new = big_win[big_win[c] <= th]
        print(f"  大亏命中: {len(loss_new)}/{len(big_loss)}, 大涨误杀: {len(win_new)}/{len(big_win)}")

        # 逐规则看重叠
        for rname, rmask in rules.items():
            if rname not in ref.columns and rname != "R6_limit_down" and rname != "R7_top10_buysm":
                continue
            # 在大亏中，新因子命中且该规则也命中的数量
            loss_r = big_loss[rmask.reindex(big_loss.index).fillna(False)]
            overlap = loss_r[loss_r[c] <= th]
            if len(overlap) > 0 or len(loss_new) > 0:
                print(f"  {rname}: 新因子单独={len(loss_new) - len(overlap)}, 与{rname}重叠={len(overlap)}, "
                      f"{rname}总命中大亏={len(loss_r)}")

    # 11. 组合规则扫描：新因子 + 现有 v3 中某因子
    print(f"\n{'='*80}")
    print("七、组合规则扫描：新因子(<=P20) AND 现有因子")
    print(f"  寻找新因子与现有因子的组合，能高精准拦截大亏")
    print(f"{'='*80}")

    combo_factors = [
        "market_liquidity_change", "mkt_dispersion_shock_v2",
        "market_advance_decline_ratio", "sector_leadership_persistence_v3",
        "ulcer_index", "momentum_peak_decay_3d", "intra_sector_momentum_rank",
        "high_position_volume_surge", "prev_day_blowoff_risk",
        "sector_momentum_accel_rank", "return_20d_intra_z",
    ]

    for c_new in NEW_FACTORS:
        if c_new not in ref.columns:
            continue
        new_vals = ref[c_new].dropna()
        if len(new_vals) < 10:
            continue

        # 新因子阈值：P20
        th_new = new_vals.quantile(0.20)

        print(f"\n--- {c_new} <= {th_new:.4f} AND existing_factor ---")

        for c_old in combo_factors:
            if c_old not in ref.columns:
                continue
            old_vals = ref[c_old].dropna()
            if len(old_vals) < 10:
                continue

            # 对现有因子扫描 3 个方向（极端值）
            for q_old in [0.20, 0.30, 0.40, 0.60, 0.70, 0.80]:
                th_old = old_vals.quantile(q_old)
                # 低方向
                mask_combo_low = (ref[c_new] <= th_new) & (ref[c_old] <= th_old)
                loss_hit = (big_loss[c_new] <= th_new) & (big_loss[c_old] <= th_old)
                win_hit = (big_win[c_new] <= th_new) & (big_win[c_old] <= th_old)
                loss_n = loss_hit.sum()
                win_n = win_hit.sum()
                if loss_n >= 2 and win_n <= max(2, len(big_win) * 0.05):
                    leak_hit = 0
                    if len(recent_leak) > 0:
                        leak_hit = ((recent_leak[c_new] <= th_new) & (recent_leak[c_old] <= th_old)).sum()
                    print(f"  & {c_old} <= {th_old:.4f} (P{q_old*100:.0f}): "
                          f"大亏={loss_n}/{len(big_loss)}, 大涨={win_n}/{len(big_win)}, "
                          f"漏网={leak_hit}/{len(recent_leak)}")

                # 高方向
                mask_combo_high = (ref[c_new] <= th_new) & (ref[c_old] >= th_old)
                loss_hit = (big_loss[c_new] <= th_new) & (big_loss[c_old] >= th_old)
                win_hit = (big_win[c_new] <= th_new) & (big_win[c_old] >= th_old)
                loss_n = loss_hit.sum()
                win_n = win_hit.sum()
                if loss_n >= 2 and win_n <= max(2, len(big_win) * 0.05):
                    leak_hit = 0
                    if len(recent_leak) > 0:
                        leak_hit = ((recent_leak[c_new] <= th_new) & (recent_leak[c_old] >= th_old)).sum()
                    print(f"  & {c_old} >= {th_old:.4f} (P{q_old*100:.0f}): "
                          f"大亏={loss_n}/{len(big_loss)}, 大涨={win_n}/{len(big_win)}, "
                          f"漏网={leak_hit}/{len(recent_leak)}")

    # 12. 最终评估
    print(f"\n{'='*80}")
    print("八、最终评估")
    print(f"{'='*80}")
    print("""
评估维度:
1. 因子分离度 (z-dist): 大亏 vs 大涨的分布差异
2. 拦截能力: 能拦截多少大亏（尤其是 v3 漏网的）
3. 误杀风险: 会误杀多少大涨
4. 互补性: 与现有 v3 规则的重叠度（新增拦截价值）
5. 组合潜力: 与现有因子组合的精准拦截能力
""")


if __name__ == "__main__":
    main()
