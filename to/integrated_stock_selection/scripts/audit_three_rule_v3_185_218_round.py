"""Audit current three_rule_v3 on the 185 sample and the new 218-round extension."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
BASE_SCRIPT = Path(__file__).with_name("audit_reg_severe_up_current_v3_185.py")

OLD_METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
OLD_SHADOW = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"
NEW_METRICS = RESULTS / "walkforward_metrics_20260611_232136.json"
NEW_SHADOW = RESULTS / "walkforward_shadow_skipped_opportunities_20260611_232136.json"

EXTENSION_START = pd.Timestamp("2026-05-15")
OUT_EVENTS = RESULTS / "three_rule_v3_185_218_round_event_audit.csv"
OUT_RULES = RESULTS / "three_rule_v3_185_218_round_rule_audit.csv"
OUT_MISSES = RESULTS / "three_rule_v3_218_round_extension_unfiltered_losses.csv"
OUT_THRESHOLD_SCAN = RESULTS / "three_rule_v3_185_218_round_threshold_neighborhood.csv"
OUT_CANDIDATE_SCAN = RESULTS / "three_rule_v3_185_218_round_new_rule_candidates.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("three_rule_base_audit", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def records_from_json(path: Path, metrics: bool) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if metrics:
        return obj[0]["all_results"]["1"]["trades"]
    records: list[dict] = []

    def walk(value) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if "stock_code" in value and "actual_return" in value:
                records.append(value)
            else:
                for item in value.values():
                    walk(item)

    walk(obj)
    return records


def normalize(records: list[dict], event_type: str, sample: str) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()
    frame["event_type"] = event_type
    frame["sample"] = sample
    date_col = frame.get("buy_datetime", frame.get("shadow_attempt_buy_date"))
    frame["buy_date"] = pd.to_datetime(date_col.astype(str).str[:10])
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["actual_return"] = pd.to_numeric(frame["actual_return"], errors="coerce")
    return frame[["sample", "event_type", "stock_code", "buy_date", "actual_return"]]


def load_events() -> pd.DataFrame:
    parts = [
        normalize(records_from_json(OLD_METRICS, True), "actual", "old_185"),
        normalize(records_from_json(OLD_SHADOW, False), "shadow", "old_185"),
        normalize(records_from_json(NEW_METRICS, True), "actual", "round_218"),
        normalize(records_from_json(NEW_SHADOW, False), "shadow", "round_218"),
    ]
    events = pd.concat(parts, ignore_index=True)
    # Prefer the newer round for overlapping dates, while retaining old-only events.
    events["sample_priority"] = events["sample"].map({"old_185": 0, "round_218": 1})
    events = events.sort_values("sample_priority").drop_duplicates(
        ["event_type", "stock_code", "buy_date", "actual_return"], keep="last"
    )
    events["cohort"] = np.where(events["buy_date"] >= EXTENSION_START, "extension", "history")
    return events.drop(columns="sample_priority").reset_index(drop=True)


def attach_factors(events: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    columns = list(dict.fromkeys(c for c in required if c in available))
    factors = pd.read_parquet(FACTORS, columns=["date", "stock_code", *columns])
    factors["date"] = pd.to_datetime(factors["date"])
    factors["stock_code"] = factors["stock_code"].astype(str)
    trading_days = np.array(sorted(factors["date"].dropna().unique()), dtype="datetime64[ns]")
    positions = np.searchsorted(
        trading_days, events["buy_date"].to_numpy(dtype="datetime64[ns]"), side="left"
    ) - 1
    events = events.copy()
    events["factor_date"] = [
        pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT for pos in positions
    ]
    factors = factors.rename(columns={"date": "factor_date"})
    return events.merge(factors, on=["factor_date", "stock_code"], how="left")


def outcome(frame: pd.DataFrame, mask: pd.Series) -> dict:
    ret = pd.to_numeric(frame["actual_return"], errors="coerce")
    return {
        "hits": int(mask.sum()),
        "big_loss": int((mask & (ret <= -0.04)).sum()),
        "small_loss": int((mask & (ret > -0.04) & (ret < 0)).sum()),
        "gain_0_8": int((mask & (ret >= 0) & (ret < 0.08)).sum()),
        "big_gain_8": int((mask & (ret >= 0.08)).sum()),
        "big_gain_10": int((mask & (ret >= 0.10)).sum()),
    }


def rule_audit(events: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_name, subset in (
        ("combined", pd.Series(True, index=events.index)),
        ("history", events["cohort"].eq("history")),
        ("extension", events["cohort"].eq("extension")),
        ("extension_actual", events["cohort"].eq("extension") & events["event_type"].eq("actual")),
        ("extension_shadow", events["cohort"].eq("extension") & events["event_type"].eq("shadow")),
    ):
        for rule in [f"r{i}" for i in range(1, 13)] + ["current_union"]:
            rows.append({"subset": subset_name, "rule": rule, **outcome(events, subset & masks[rule])})
    return pd.DataFrame(rows)


def threshold_neighborhood(events: pd.DataFrame, base) -> pd.DataFrame:
    """Small, interpretable neighborhoods for existing two-factor rules."""
    z = base.z
    specs = {
        "r1": ("runup_ma5_exhaustion", ">", [0.6, 0.7, 0.8, 0.9],
               "cpv_factor", "<=", [-0.6, -0.5, -0.4, -0.3]),
        "r3": ("sector_leadership_persistence_v3", "<=", [-0.30, -0.24, -0.18, -0.12],
               "return_20d_intra_z", "<=", [0.2, 0.4, 0.6, 0.8]),
        "r5": ("mkt_dispersion_shock_v2", "<", [0.90, 0.94, 0.98, 1.02],
               "market_advance_decline_ratio", "<", [0.30, 0.34, 0.38, 0.42]),
        "r7": ("top10_10d_strength_level_v3", ">=", [0.60, 0.65, 0.70, 0.75],
               "buy_sm_amount_rate", "<=", [-1.50, -1.25, -1.00, -0.75]),
        "r8": ("prev_non_first_limit_up_performance_percentile_20d", ">=", [0.70, 0.75, 0.80, 0.85],
               "kdj_j_9", "<=", [0.50, 0.75, 1.00, 1.25]),
        "r10": ("reg_severe_up_10d_ratio_pct", ">=", [0.95, 0.97, 0.98, 0.99],
                "market_advance_decline_ratio", ">=", [3.5, 4.0, 4.5, 5.0]),
        "r11": ("price_ma5_ratio_raw", ">=", [7.0, 8.0, 9.0, 10.0],
                "ma_convergence_score", ">=", [0.15, 0.20, 0.25, 0.30]),
        "r12": ("ma5_pullback_depth_10d", "<=", [-0.04, -0.05, -0.06, -0.07],
                "ret_3d_vs_10d_accel", ">=", [5.0, 5.5, 6.0, 6.5]),
    }

    def compare(values: pd.Series, op: str, threshold: float) -> pd.Series:
        return {
            ">": values > threshold,
            ">=": values >= threshold,
            "<": values < threshold,
            "<=": values <= threshold,
        }[op]

    rows = []
    for rule, (a, op_a, levels_a, b, op_b, levels_b) in specs.items():
        for ta in levels_a:
            for tb in levels_b:
                mask = compare(z(events, a), op_a, ta) & compare(z(events, b), op_b, tb)
                row = {"rule": rule, "threshold_a": ta, "threshold_b": tb}
                for subset_name, subset in (
                    ("history", events["cohort"].eq("history")),
                    ("extension", events["cohort"].eq("extension")),
                    ("extension_actual", events["cohort"].eq("extension") & events["event_type"].eq("actual")),
                    ("extension_shadow", events["cohort"].eq("extension") & events["event_type"].eq("shadow")),
                ):
                    for key, value in outcome(events, mask & subset).items():
                        row[f"{subset_name}_{key}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def candidate_rule_scan(events: pd.DataFrame, masks: pd.DataFrame, base) -> pd.DataFrame:
    """Scan a small set of financially interpretable rules suggested by extension misses."""
    z = base.z
    union = masks["current_union"]
    families: list[tuple[str, pd.Series]] = []

    pullback = z(events, "ma5_pullback_depth_10d")
    path = z(events, "runup_path_turbulence_10d")
    churn = z(events, "high_level_churn_risk_10d")
    accel = z(events, "ret_3d_vs_10d_accel")
    reg_pct = z(events, "reg_severe_up_10d_ratio_pct")
    adr = z(events, "market_advance_decline_ratio")
    liq = z(events, "market_liquidity_change")
    upper = z(events, "upper_shadow_pressure_5d")
    mom_peak = z(events, "momentum_peak_decay_3d")
    ulcer = z(events, "ulcer_index")
    top10 = z(events, "top10_10d_strength_level_v3")
    buy_sm = z(events, "buy_sm_amount_rate")
    sector_rank = z(events, "intra_sector_momentum_rank")

    for p in [-0.04, -0.05, -0.06, -0.08, -0.10]:
        for t in [3.0, 4.0, 5.0, 6.0]:
            families.append((f"pullback<={p:.2f}&path_turb>={t:.1f}", (pullback <= p) & (path >= t)))
        for a in [0.0, -0.5, -1.0, -2.0, -4.0]:
            families.append((f"pullback<={p:.2f}&accel<={a:.1f}", (pullback <= p) & (accel <= a)))
        for c in [3.0, 4.0, 5.0, 6.0]:
            families.append((f"pullback<={p:.2f}&churn>={c:.1f}", (pullback <= p) & (churn >= c)))

    for t in [3.0, 4.0, 5.0, 6.0]:
        for r in [0.95, 0.97, 0.98, 0.99]:
            families.append((f"path_turb>={t:.1f}&reg_pct>={r:.2f}", (path >= t) & (reg_pct >= r)))
        for a in [0.30, 0.50, 1.00, 2.00]:
            families.append((f"path_turb>={t:.1f}&adr<={a:.2f}", (path >= t) & (adr <= a)))
        for l in [0.94, 0.98, 1.00, 1.02]:
            families.append((f"path_turb>={t:.1f}&liq<={l:.2f}", (path >= t) & (liq <= l)))
        for u in [1.0, 2.0, 3.0, 4.0]:
            families.append((f"path_turb>={t:.1f}&upper>={u:.1f}", (path >= t) & (upper >= u)))
        for m in [1.0, 2.0, 3.0, 4.0]:
            families.append((f"path_turb>={t:.1f}&mom_peak>={m:.1f}", (path >= t) & (mom_peak >= m)))
        for u in [-1.0, -2.0, -3.0, -4.0]:
            families.append((f"path_turb>={t:.1f}&ulcer<={u:.1f}", (path >= t) & (ulcer <= u)))
        for s in [0.67, 0.75, 0.85, 0.95]:
            families.append((f"path_turb>={t:.1f}&sector_rank>={s:.2f}", (path >= t) & (sector_rank >= s)))

    # Complementary R10 paths: severe-up pressure under non-euphoric breadth.
    for r in [0.97, 0.98, 0.99, 0.995]:
        for a in [0.5, 1.0, 1.5, 2.0, 2.5]:
            families.append((f"reg_pct>={r:.3f}&adr<={a:.1f}", (reg_pct >= r) & (adr <= a)))
        for t in [3.0, 4.0, 5.0, 6.0]:
            families.append((f"reg_pct>={r:.3f}&path_turb>={t:.1f}", (reg_pct >= r) & (path >= t)))
        for s in [0.10, 0.25, 0.45, 0.65]:
            families.append((f"reg_pct>={r:.3f}&top10<={s:.2f}", (reg_pct >= r) & (top10 <= s)))
        for b in [-1.5, -1.0, -0.5, 0.0]:
            families.append((f"reg_pct>={r:.3f}&buy_sm<={b:.1f}", (reg_pct >= r) & (buy_sm <= b)))

    rows = []
    for label, standalone in families:
        incremental = standalone & ~union
        row = {"rule": label}
        for subset_name, subset in (
            ("history", events["cohort"].eq("history")),
            ("extension", events["cohort"].eq("extension")),
            ("extension_actual", events["cohort"].eq("extension") & events["event_type"].eq("actual")),
            ("extension_shadow", events["cohort"].eq("extension") & events["event_type"].eq("shadow")),
        ):
            for prefix, mask in (("standalone", standalone & subset), ("incremental", incremental & subset)):
                for key, value in outcome(events, mask).items():
                    row[f"{subset_name}_{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    base = load_base_module()
    required = list(dict.fromkeys([
        *base.FACTOR_COLS,
        "score_z_in_day",
        "high_level_churn_risk_10d",
        "runup_path_turbulence_10d",
    ]))
    events = attach_factors(load_events(), required)
    # Runtime score_z is unavailable for the new round; R9 is consequently false there.
    masks = base.current_v3(events)
    for col in masks.columns:
        events[col] = masks[col].to_numpy()
    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")

    rules = rule_audit(events, masks)
    rules.to_csv(OUT_RULES, index=False, encoding="utf-8-sig")

    misses = events[
        events["cohort"].eq("extension")
        & (events["actual_return"] < 0)
        & ~events["current_union"]
    ].sort_values("actual_return")
    misses.to_csv(OUT_MISSES, index=False, encoding="utf-8-sig")

    threshold_scan = threshold_neighborhood(events, base)
    threshold_scan.to_csv(OUT_THRESHOLD_SCAN, index=False, encoding="utf-8-sig")
    candidate_scan = candidate_rule_scan(events, masks, base)
    candidate_scan.to_csv(OUT_CANDIDATE_SCAN, index=False, encoding="utf-8-sig")

    print("EXTENSION RULE AUDIT")
    print(rules[rules["subset"].str.startswith("extension")].to_string(index=False))
    print("\nEXTENSION UNFILTERED LOSSES")
    cols = ["event_type", "buy_date", "stock_code", "actual_return", "factor_date"]
    print(misses[cols].to_string(index=False))
    print(f"\nevents={OUT_EVENTS}")
    print(f"rules={OUT_RULES}")
    print(f"misses={OUT_MISSES}")
    print(f"threshold_scan={OUT_THRESHOLD_SCAN}")
    print(f"candidate_scan={OUT_CANDIDATE_SCAN}")


if __name__ == "__main__":
    main()
