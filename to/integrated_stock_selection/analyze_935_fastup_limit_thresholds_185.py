from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DETAIL = RESULTS / "open_935_return_relationship_185_detail.csv"


def summarize(name, mask, df):
    hit = df[mask].copy()
    miss = df[~mask].copy()

    def part_stats(part, prefix):
        if part.empty:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_avg_ret": np.nan,
                f"{prefix}_median_ret": np.nan,
                f"{prefix}_big_loss_rate": np.nan,
                f"{prefix}_big_win_rate": np.nan,
                f"{prefix}_sum_pnl": 0.0,
            }
        return {
            f"{prefix}_n": len(part),
            f"{prefix}_avg_ret": part["final_return"].mean() * 100,
            f"{prefix}_median_ret": part["final_return"].median() * 100,
            f"{prefix}_big_loss_rate": part["is_big_loss"].mean() * 100,
            f"{prefix}_big_win_rate": part["is_big_win"].mean() * 100,
            f"{prefix}_sum_pnl": part["profit_loss"].sum(),
        }

    rec = {"rule": name}
    rec.update(part_stats(hit, "hit"))
    rec.update(part_stats(miss, "miss"))
    rec["half_hit_delta_pnl_if_halved"] = -0.5 * hit["profit_loss"].sum()
    rec["hit_big_loss_count"] = int(hit["is_big_loss"].sum())
    rec["hit_big_win_count"] = int(hit["is_big_win"].sum())
    return rec


def main():
    df = pd.read_csv(DETAIL)
    for c in [
        "open_gap",
        "ret_0935_vs_open",
        "high_0935_vs_open",
        "low_0935_vs_open",
        "buy_vs_open",
        "final_return",
        "profit_loss",
        "prev_close_5m",
        "open_5m",
        "close_0935",
        "high_0935",
        "low_0935",
        "buy_price",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # First 5-minute K-line path features.
    df["high_0935_vs_prev_close"] = df["high_0935"] / df["prev_close_5m"] - 1.0
    df["close_0935_vs_prev_close"] = df["close_0935"] / df["prev_close_5m"] - 1.0
    df["close_0935_vs_buy"] = df["close_0935"] / df["buy_price"] - 1.0
    df["high_0935_vs_buy"] = df["high_0935"] / df["buy_price"] - 1.0
    df["upper_pullback_from_high_open"] = (df["high_0935"] - df["close_0935"]) / df["open_5m"]
    df["upper_retrace_ratio"] = np.where(
        df["high_0935"] > df["open_5m"],
        (df["high_0935"] - df["close_0935"]) / (df["high_0935"] - df["open_5m"]),
        np.nan,
    )
    df["close_position_in_first_bar"] = np.where(
        df["high_0935"] > df["low_0935"],
        (df["close_0935"] - df["low_0935"]) / (df["high_0935"] - df["low_0935"]),
        np.nan,
    )
    df["fast_up_first5"] = df["high_0935_vs_open"] >= 0.015
    df["fast_up_close_strong"] = df["ret_0935_vs_open"] >= 0.015
    df["spike_fade_close_below_open"] = (df["high_0935_vs_open"] >= 0.015) & (df["ret_0935_vs_open"] <= 0)
    df["spike_fade_upper_1pct"] = (df["high_0935_vs_open"] >= 0.02) & (df["upper_pullback_from_high_open"] >= 0.01)
    df["spike_fade_retrace_50"] = (df["high_0935_vs_open"] >= 0.02) & (df["upper_retrace_ratio"] >= 0.5)
    df["near_limit_0935_prev_9p5"] = df["close_0935_vs_prev_close"] >= 0.095
    df["near_limit_0935_prev_19p0"] = df["close_0935_vs_prev_close"] >= 0.19

    df.to_csv(RESULTS / "open_935_fastup_limit_thresholds_185_detail.csv", index=False, encoding="utf-8-sig")

    corr_cols = [
        "high_0935_vs_open",
        "ret_0935_vs_open",
        "upper_pullback_from_high_open",
        "upper_retrace_ratio",
        "close_position_in_first_bar",
        "close_0935_vs_prev_close",
        "close_0935_vs_buy",
        "high_0935_vs_prev_close",
        "high_0935_vs_buy",
        "final_return",
    ]
    corr = df[corr_cols].corr(method="spearman")
    corr.to_csv(RESULTS / "open_935_fastup_limit_thresholds_185_spearman_corr.csv", encoding="utf-8-sig")

    rule_rows = []
    for name in [
        "fast_up_first5",
        "fast_up_close_strong",
        "spike_fade_close_below_open",
        "spike_fade_upper_1pct",
        "spike_fade_retrace_50",
        "near_limit_0935_prev_9p5",
        "near_limit_0935_prev_19p0",
    ]:
        rule_rows.append(summarize(name, df[name].fillna(False), df))

    # 9:35 add-limit scan. These rules mean: among close>=open candidates, allow adding
    # the second half only when both optional caps are satisfied.
    base_add = df["ret_0935_vs_open"] >= 0
    prev_caps = [0.03, 0.05, 0.07, 0.095, 0.12, 0.15, 0.19]
    buy_caps = [0.015, 0.03, 0.05, 0.07, 0.10, 0.15]
    for cap in prev_caps:
        rule_rows.append(summarize(f"add935_close_ge_open_and_close_vs_prev_le_{cap:.3f}", base_add & (df["close_0935_vs_prev_close"] <= cap), df))
    for cap in buy_caps:
        rule_rows.append(summarize(f"add935_close_ge_open_and_close_vs_buy_le_{cap:.3f}", base_add & (df["close_0935_vs_buy"] <= cap), df))
    for pc in prev_caps:
        for bc in buy_caps:
            rule_rows.append(
                summarize(
                    f"add935_close_ge_open_prev_le_{pc:.3f}_buy_le_{bc:.3f}",
                    base_add & (df["close_0935_vs_prev_close"] <= pc) & (df["close_0935_vs_buy"] <= bc),
                    df,
                )
            )

    rules = pd.DataFrame(rule_rows)
    rules.to_csv(RESULTS / "open_935_fastup_limit_thresholds_185_rule_stats.csv", index=False, encoding="utf-8-sig")

    # Bin tables for limit candidates.
    df["close_vs_prev_bin"] = pd.cut(
        df["close_0935_vs_prev_close"],
        bins=[-np.inf, -0.03, -0.01, 0.0, 0.03, 0.05, 0.07, 0.095, 0.12, 0.19, np.inf],
        labels=["<=-3%", "-3~-1%", "-1~0%", "0~3%", "3~5%", "5~7%", "7~9.5%", "9.5~12%", "12~19%", ">19%"],
    )
    df["close_vs_buy_bin"] = pd.cut(
        df["close_0935_vs_buy"],
        bins=[-np.inf, -0.03, -0.01, 0.0, 0.015, 0.03, 0.05, 0.07, 0.10, 0.15, np.inf],
        labels=["<=-3%", "-3~-1%", "-1~0%", "0~1.5%", "1.5~3%", "3~5%", "5~7%", "7~10%", "10~15%", ">15%"],
    )
    bin_rows = []
    for col in ["close_vs_prev_bin", "close_vs_buy_bin"]:
        g = df.groupby(col, dropna=False, observed=False)
        x = g.agg(
            n=("stock_code", "size"),
            avg_final_ret=("final_return", "mean"),
            median_final_ret=("final_return", "median"),
            big_loss_rate=("is_big_loss", "mean"),
            big_win_rate=("is_big_win", "mean"),
            sum_pnl=("profit_loss", "sum"),
        ).reset_index()
        x.insert(0, "bin_type", col)
        for c in ["avg_final_ret", "median_final_ret", "big_loss_rate", "big_win_rate"]:
            x[c] = x[c] * 100
        bin_rows.append(x)
    bins = pd.concat(bin_rows, ignore_index=True, sort=False)
    bins.to_csv(RESULTS / "open_935_fastup_limit_thresholds_185_bins.csv", index=False, encoding="utf-8-sig")

    print("rows", len(df), "actual", int((df["source"] == "actual").sum()), "shadow", int((df["source"] == "shadow").sum()))
    print("\n== spearman with final_return ==")
    print(corr["final_return"].sort_values().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n== path pattern rules ==")
    print(rules.head(7).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== best add-limit rules by hit avg return, n>=30 ==")
    add_rules = rules[rules["rule"].str.startswith("add935") & (rules["hit_n"] >= 30)].copy()
    cols = ["rule", "hit_n", "hit_avg_ret", "hit_median_ret", "hit_big_loss_rate", "hit_big_win_rate", "hit_sum_pnl", "miss_avg_ret", "miss_sum_pnl"]
    print(add_rules.sort_values(["hit_avg_ret", "hit_sum_pnl"], ascending=[False, False]).head(25)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== bins ==")
    print(bins.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
