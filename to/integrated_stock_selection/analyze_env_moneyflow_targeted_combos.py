import pandas as pd

import analyze_moneyflow_path_factor_combos as base_analysis


ENV_FACTORS = [
    "market_60d_new_high_ratio",
    "top10_10d_today_return_level_v3",
    "top10_3d_recent3_strength_v3",
    "top10_5d_recent3_strength_v3",
    "top10_10d_recent3_strength_v3",
    "top10_10d_strength_level_v3",
]

MONEYFLOW_FACTORS = [
    ("buy_sm_amount_rate", "low"),
    ("lg_pos_sm_neg_10d_days", "high"),
    ("price_up_lg_sm_divergence_10d", "high"),
    ("price_up_sm_outflow_10d", "high"),
    ("sm_outflow_strength_10d", "high"),
    ("sm_outflow_lg_sm_divergence_10d", "high"),
    ("moneyflow_divergence_10d", "high"),
    ("lg_minus_sm_10d_mean", "high"),
]


def load_frame():
    events = base_analysis.load_events()
    factors, numeric_cols = base_analysis.load_factor_frame()
    base = base_analysis.attach_factor_snapshot(events, factors, numeric_cols)
    df = base.merge(
        base_analysis.load_moneyflow_features(),
        on="key",
        how="left",
        suffixes=("", "_mf"),
    )
    df["current_v3"] = base_analysis.current_v3_mask(df)
    return df, base_analysis.make_groups(df)


def thresholds(feature, direction, recent):
    s = pd.to_numeric(recent[feature], errors="coerce").dropna()
    qs = [0.0, 0.25, 0.50, 0.75] if direction == "high" else [0.25, 0.50, 0.75, 1.0]
    return sorted(set(float(s.quantile(q)) for q in qs))


def counts(combo, recent, wins, losses, neutral):
    return (
        int(base_analysis.combo_mask(recent, combo).sum()),
        int(base_analysis.combo_mask(wins, combo).sum()),
        int(base_analysis.combo_mask(losses, combo).sum()),
        int(base_analysis.combo_mask(neutral, combo).sum()),
    )


def rule_text(combo):
    parts = []
    for feature, direction, threshold, _pct in combo:
        op = ">=" if direction == "high" else "<="
        parts.append(f"{feature} {op} {threshold:.4g}")
    return " & ".join(parts)


def add_result(rows, kind, combo, recent, wins, losses, neutral):
    r, w, l, n = counts(combo, recent, wins, losses, neutral)
    if r < 2:
        return
    score = r * 14 + l * 2 - w * 10 - n
    if kind == "triple":
        score -= 3
    rows.append(
        {
            "kind": kind,
            "recent_hits": r,
            "win_hits": w,
            "hist_loss_hits": l,
            "neutral_hits": n,
            "score": score,
            "rule": rule_text(combo),
            "features": "|".join(c[0] for c in combo),
            "combo": combo,
        }
    )


def main():
    df, groups = load_frame()
    recent, wins, losses, neutral, _hist = groups

    rows = []
    for env in ENV_FACTORS:
        for mf, mf_dir in MONEYFLOW_FACTORS:
            for env_th in thresholds(env, "high", recent):
                for mf_th in thresholds(mf, mf_dir, recent):
                    combo = [(env, "high", env_th, 0), (mf, mf_dir, mf_th, 0)]
                    add_result(rows, "pair", combo, recent, wins, losses, neutral)

    for env in ENV_FACTORS:
        for mf, mf_dir in MONEYFLOW_FACTORS[1:]:
            for env_th in thresholds(env, "high", recent):
                for buy_sm_th in thresholds("buy_sm_amount_rate", "low", recent):
                    for mf_th in thresholds(mf, mf_dir, recent):
                        combo = [
                            (env, "high", env_th, 0),
                            ("buy_sm_amount_rate", "low", buy_sm_th, 0),
                            (mf, mf_dir, mf_th, 0),
                        ]
                        add_result(rows, "triple", combo, recent, wins, losses, neutral)

    out = pd.DataFrame(rows).sort_values(
        ["score", "recent_hits", "win_hits", "neutral_hits", "hist_loss_hits"],
        ascending=[False, False, True, True, False],
    )
    out.drop(columns=["combo"]).to_csv(
        base_analysis.RESULTS / "env_moneyflow_combo_targeted_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline = [
        ("top10_10d_strength_level_v3", "high", 0.70, 0),
        ("buy_sm_amount_rate", "low", -1.25, 0),
    ]
    print("== Baseline ==")
    print(rule_text(baseline), counts(baseline, recent, wins, losses, neutral))
    for ds in sorted(df["dataset"].unique()):
        d = df[df["dataset"].eq(ds)]
        print(ds, base_analysis.summarize(d, base_analysis.combo_mask(d, baseline)))

    print("\n== Top targeted env-moneyflow combos ==")
    print(out.drop(columns=["combo"]).head(60).to_string(index=False))

    print("\n== Top 12 details ==")
    for _, row in out.head(12).iterrows():
        combo = row["combo"]
        print(f"\n-- {row['rule']} --")
        print("unique R/W/L/N:", counts(combo, recent, wins, losses, neutral))
        for ds in sorted(df["dataset"].unique()):
            d = df[df["dataset"].eq(ds)]
            print(ds, base_analysis.summarize(d, base_analysis.combo_mask(d, combo)))


if __name__ == "__main__":
    main()
