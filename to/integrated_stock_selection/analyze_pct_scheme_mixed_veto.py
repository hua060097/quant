from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_moneyflow_veto as mfmod
import analyze_raw_pct_hybrid_and_failures as rawmod
import analyze_selected_factor_fixed_raw_vs_pct as sfmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

SCHEME_NAME = "switch_reg_pct_mktstate_hi95_lo85"


def summarize(series: pd.Series) -> tuple[float, float, float, float]:
    traded = pd.to_numeric(series, errors="coerce").dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        am.calc_mdd(traded),
        float((traded > 0).mean()),
    )


def split_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.sort_values("date").reset_index(drop=True)
    n = len(work)
    half = n // 2
    fifth = n // 5 if n >= 5 else n
    recent_n = min(13, n)
    return {
        "full": work,
        "first_half": work.iloc[:half],
        "second_half": work.iloc[half:],
        "seg5_last": work.iloc[-fifth:] if fifth > 0 else work.iloc[0:0],
        f"recent_{recent_n}d": work.iloc[-recent_n:],
    }


def build_pct_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct_thr = rawmod.build_pct_state_threshold(df, 0.95, 0.85)
    top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
    choose_top1 = top1_pct < pct_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def build_selected_feature_table(df: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    raw_feature_df, pct_feature_df = sfmod.load_feature_tables()
    selected = sfmod.attach_selected_features(df, rank, raw_feature_df, pct_feature_df)
    return selected


def build_technical_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    rules: dict[str, pd.Series] = {}

    # 之前最强/最稳的一些技术 veto，尽量保持金融含义清晰
    rules["tech_rawfix_cpv1p5_h2c_m0p015"] = (
        (pd.to_numeric(df["cpv_factor"], errors="coerce") >= 1.5)
        & (pd.to_numeric(df["high_to_close_drawdown"], errors="coerce") <= -0.015)
    )

    h2c_q15 = am.expanding_threshold(pd.to_numeric(df["high_to_close_drawdown"], errors="coerce"), 0.15)
    ret10_q20 = am.expanding_threshold(pd.to_numeric(df["return_10d_sector_mean"], errors="coerce"), 0.20)
    rules["tech_rawexp_h2c_q15_ret10_q20"] = (
        (pd.to_numeric(df["high_to_close_drawdown"], errors="coerce") <= h2c_q15)
        & (pd.to_numeric(df["return_10d_sector_mean"], errors="coerce") <= ret10_q20)
        & h2c_q15.notna()
        & ret10_q20.notna()
    )

    ret10_q40 = am.expanding_threshold(pd.to_numeric(df["return_10d_sector_mean"], errors="coerce"), 0.40)
    shadow_q85 = am.expanding_threshold(pd.to_numeric(df["shadow_dominance"], errors="coerce"), 0.85)
    rules["tech_rawexp_ret10_q40_shadow_q85"] = (
        (pd.to_numeric(df["return_10d_sector_mean"], errors="coerce") <= ret10_q40)
        & (pd.to_numeric(df["shadow_dominance"], errors="coerce") >= shadow_q85)
        & ret10_q40.notna()
        & shadow_q85.notna()
    )

    rules["tech_pct_cpv95_h2c20"] = (
        (pd.to_numeric(df["cpv_factor__pct"], errors="coerce") >= 0.95)
        & (pd.to_numeric(df["high_to_close_drawdown__pct"], errors="coerce") <= 0.20)
    )

    return rules


def build_moneyflow_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    lg_rate = pd.to_numeric(df["buy_lg_amount_rate"], errors="coerce")
    d5_pct = pd.to_numeric(df["net_d5_amount__pct"], errors="coerce")
    sm_rate = pd.to_numeric(df["buy_sm_amount_rate"], errors="coerce")
    net_amt = pd.to_numeric(df["net_amount"], errors="coerce")
    day_pct = pd.to_numeric(df["pct_change"], errors="coerce")

    return {
        "mf_lg_m2_sm_p2": (lg_rate <= -2.0) & (sm_rate >= 2.0),
        "mf_lg_m1_d5pct20": (lg_rate <= -1.0) & (d5_pct <= 0.20),
        "mf_lg_m2_net0_up": (lg_rate <= -2.0) & (net_amt <= 0) & (day_pct >= 2.0),
    }


def main() -> None:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    moneyflow_df = mfmod.load_moneyflow()
    rows = []

    for json_path in am.JSON_PATHS:
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        df = am.build_base_df(json_path, factor_df)
        gap_map = am.build_buy_gap_map(df)

        rank, ret = build_pct_scheme(df)
        selected_feat = build_selected_feature_table(df, rank)
        work = mfmod.attach_selected_moneyflow(df, rank, ret, gap_map, moneyflow_df)

        # merge technical columns from selected_feat
        merge_cols = ["date", "selected_code"] + [
            "cpv_factor",
            "ppo_strength",
            "return_10d_sector_mean",
            "shadow_dominance",
            "high_to_close_drawdown",
            "cpv_factor__pct",
            "ppo_strength__pct",
            "return_10d_sector_mean__pct",
            "shadow_dominance__pct",
            "high_to_close_drawdown__pct",
        ]
        work = work.merge(selected_feat[merge_cols], on=["date", "selected_code"], how="left")

        tech_rules = build_technical_rules(work)
        mf_rules = build_moneyflow_rules(work)

        all_rules = [("baseline", pd.Series(False, index=work.index), "baseline", 0.0)]

        for name, mask in tech_rules.items():
            rate = float(mask.fillna(False).mean())
            if 0.01 <= rate <= 0.35:
                all_rules.append((name, mask, "tech_only", rate))

        for name, mask in mf_rules.items():
            rate = float(mask.fillna(False).mean())
            if 0.01 <= rate <= 0.35:
                all_rules.append((name, mask, "moneyflow_only", rate))

        for mf_name, mf_mask in mf_rules.items():
            for tech_name, tech_mask in tech_rules.items():
                mask = mf_mask & tech_mask
                rate = float(mask.fillna(False).mean())
                if 0.005 <= rate <= 0.25:
                    all_rules.append((f"{mf_name}__and__{tech_name}", mask, "mixed_pair", rate))

        for veto_name, veto_mask, veto_type, veto_rate in all_rules:
            veto_ret = pd.to_numeric(work["filtered_ret"], errors="coerce").mask(veto_mask.fillna(False))
            temp = work.copy()
            temp["veto_ret"] = veto_ret
            for seg_name, seg_df in split_segments(temp).items():
                mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["veto_ret"])
                rows.append(
                    {
                        "json_name": json_path.name,
                        "label_version": label_version,
                        "scheme": SCHEME_NAME,
                        "veto": veto_name,
                        "veto_type": veto_type,
                        "segment": seg_name,
                        "active_days": int(seg_df["veto_ret"].notna().sum()),
                        "trade_rate": float(seg_df["veto_ret"].notna().mean()),
                        "mean_ret": mean_ret,
                        "cum_ret": cum_ret,
                        "max_drawdown": mdd,
                        "win_rate": win_rate,
                        "veto_rate": veto_rate,
                    }
                )

    detailed = pd.DataFrame(rows)
    detailed_path = RESULTS_DIR / "pct_scheme_mixed_veto_compare_20260419.csv"
    detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    agg = (
        detailed.groupby(["veto", "veto_type", "segment"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_veto_rate=("veto_rate", "mean"),
        )
        .sort_values(["segment", "avg_cum_ret"], ascending=[True, False])
        .reset_index(drop=True)
    )
    agg_path = RESULTS_DIR / "pct_scheme_mixed_veto_compare_aggregate_20260419.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for (veto, veto_type), sub in agg.groupby(["veto", "veto_type"], sort=False):
        row = {"veto": veto, "veto_type": veto_type}
        for _, r in sub.iterrows():
            row[f"cum__{r['segment']}"] = r["avg_cum_ret"]
            row[f"mdd__{r['segment']}"] = r["avg_mdd"]
            row[f"trade__{r['segment']}"] = r["avg_trade_rate"]
        row["score"] = (
            row.get("cum__second_half", 0) * 3.0
            + row.get("cum__seg5_last", 0) * 4.0
            + row.get("cum__recent_13d", 0) * 3.0
            + row.get("cum__full", 0) * 0.5
            + row.get("mdd__full", 0) * 0.5
        )
        score_rows.append(row)
    ranked = pd.DataFrame(score_rows).sort_values("score", ascending=False).reset_index(drop=True)
    ranked_path = RESULTS_DIR / "pct_scheme_mixed_veto_ranked_20260419.csv"
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")

    print(f"[OK] detailed -> {detailed_path}")
    print(f"[OK] aggregate -> {agg_path}")
    print(f"[OK] ranked -> {ranked_path}")


if __name__ == "__main__":
    main()
