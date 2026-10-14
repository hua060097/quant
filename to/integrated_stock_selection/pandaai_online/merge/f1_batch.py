# -*- coding: utf-8 -*-
"""F1 批量： market 门控用掩码捷径（已证明等价），sector 门控用全重算。
用法: python f1_batch.py --cands cand_w0.json --out f1_w0.csv --tag w0
cands JSON: [{name, base, gate, gate_type(market/sector), s, d}]
输出每行: H3 全部原始统计量（§8.4），失败原因记入 fail_reason。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B
import eval_v2 as E
import h3_eval as H
import windows as W

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
# 搜索窗口改为单一事实源（windows.py）。原为硬编码 "2024-12-26"，即 P0/P1
# 时期的旧窗口；现默认生产版口径 signal≤2026-09-11 / labels≤2026-09-18。
# 复现旧结果请显式传 --label-max 2024-12-26。
LABEL_MAX = W.FACTOR_SCREEN_MAX
N_PERM = 200


# 门控文件路径（默认 = 搜索空间产出的市场候选池；2026-10-12 起旧手工门控
# `gates/market_gates.parquet` 已删除——门控定义由搜索生成）
GATES_PARQUET = os.path.join(RESULTS, "gates_search", "market_gates_search.parquet")


def load_gates():
    mg = pd.read_parquet(GATES_PARQUET)
    mg["date"] = pd.to_datetime(mg["date"])
    return mg


def run_one_market(m, base_col, d, gate_series, s, uniq_dates, ic_z):
    """掩码捷径：on-day 上 F1 与 d·x 秩相同（已证明），只置换掩码。"""
    g = np.asarray(gate_series, dtype=float)
    on = np.isfinite(g) & (s * g > 0)
    f1_ic = np.where(on, ic_z, np.nan)

    def fn(gp):
        gp = np.asarray(gp, dtype=float)
        o2 = np.isfinite(gp) & (s * gp > 0)
        return np.where(o2, ic_z, np.nan)

    res = {"delta": H.delta_ic(np.asarray(uniq_dates), f1_ic, ic_z)}
    res["slope"] = H.slope_test(ic_z, g)
    res["placebo"] = H.block_placebo(fn, g, ic_z, n_perm=N_PERM, seed=0)
    # spread：F1 在 on-day 的 spread = base spread；off-day NaN
    res["_f1_ic"] = f1_ic
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="f1")
    ap.add_argument("--label-max", default=LABEL_MAX,
                    help=f"信号日上界（默认选择窗 {W.FACTOR_SCREEN_MAX}；"
                         f"复现 P0/P1 旧结果用 {W.LEGACY_SEARCH_END}）")
    ap.add_argument("--form", default="g", choices=("g", "t"),
                    help="门控组合形式：g=连续(2p-1，on 为 g>0)；"
                         "t=THRESH(p>=0.7 为 on，中间弃权)。见 PLAN §16.23")
    ap.add_argument("--gates-parquet", default=None,
                    help="市场门控 parquet 路径（默认 gates_search/market_gates_search.parquet）；"
                         "指向其它候选池时用于批量评估新门控")
    a = ap.parse_args()
    label_max = a.label_max
    if a.gates_parquet:
        global GATES_PARQUET
        GATES_PARQUET = a.gates_parquet
        print(f"[{a.tag}] gates parquet = {a.gates_parquet}", flush=True)
    win = {"search_start": W.FACTORS_FIRST_SIGNAL_DATE, "search_end": label_max,
           "label_available_max": W.LABEL_MAX_DATE, "evidence": "E1"}
    print(f"[{a.tag}] window {win['search_start']}~{win['search_end']} "
          f"(labels≤{win['label_available_max']})", flush=True)
    cands = json.load(open(a.cands, encoding="utf-8"))
    print(f"[{a.tag}] candidates={len(cands)}", flush=True)
    # 本批只跑 market 门控（掩码捷径，已证明等价）。
    # sector 门控需全重算（逐置换重排），独立第二批实现。
    sc = [c for c in cands if c["gate_type"] != "market"]
    assert not sc, f"本批只接受 market 候选，sector 候选 {len(sc)} 个请走第二批"
    mk = cands
    # labels
    lab = B.read_labels(["r3", "y_prod", "tradable"], label_max=label_max, label=f"f1-{a.tag}")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= label_max)].copy()
    # gates
    mg = load_gates()
    # base 面板（去重列）
    base_cols = sorted({c["base"] for c in cands})
    fac = B.read_factors(base_cols, data_max=W.LABEL_MAX_DATE, label=f"f1-{a.tag}")
    fac["date"] = pd.to_datetime(fac["date"])
    m = fac.merge(lab[["stock_code", "date", "r3", "y_prod"]], on=["stock_code", "date"], how="inner")
    m = m[m["date"] <= label_max].copy()
    del fac
    print(f"[{a.tag}] panel rows={len(m)}", flush=True)
    # 方向
    b20 = {b["factor"]: b["direction"] for b in
           json.load(open(os.path.join(RESULTS, "base20.json"), encoding="utf-8"))["base20"]}
    # 预计算每 base 的 IC_z（d·x）日序列
    day, dates_u, _, _, nd = E._prepared(m, base_cols[0], "r3")
    uniq = np.unique(m["date"].to_numpy())
    ic_cache = {}
    for bcol in base_cols:
        d = b20.get(bcol, 1)
        _, _, xr, yr, _ = E._prepared(m.assign(_z=pd.to_numeric(m[bcol], errors="coerce") * d), "_z", "r3")
        ic_cache[bcol] = E._daily_corr(xr, yr, day, nd)
    print(f"[{a.tag}] base IC cached", flush=True)

    started_empty = not os.path.exists(a.out)
    done_n = 0
    for i, c in enumerate(mk):
        t1 = time.time()
        rec = {"name": c["name"], "base": c["base"], "gate": c["gate"],
               "gate_type": c["gate_type"], "s": c["s"], "d": c["d"],
               "form": a.form,
               "search_start": win["search_start"], "search_end": win["search_end"],
               "label_available_max": win["label_available_max"],
               "evidence": win["evidence"]}
        try:
            gmap = dict(zip(pd.to_datetime(mg["date"]).dt.date.astype(str),
                            mg[c["gate"] + "_" + a.form].to_numpy()))
            g = np.array([gmap.get(str(pd.Timestamp(dd).date()), np.nan) for dd in uniq])
            r = run_one_market(m, c["base"], c["d"], g, c["s"], uniq, ic_cache[c["base"]])
            r.pop("_f1_ic", None)
            dl = r["delta"]
            pl = r["placebo"]
            sl = r["slope"]
            rec.update({
                "delta_mean": dl.get("mean"), "delta_ci_lo": dl.get("ci_lo"),
                "delta_ci_hi": dl.get("ci_hi"), "nblocks_pos": dl.get("nblocks_pos"),
                "blocks": json.dumps(dl.get("blocks"), ensure_ascii=False),
                "slope_b": sl.get("b"), "slope_t": sl.get("t"),
                "placebo_p": pl.get("p"), "placebo_obs": pl.get("obs"),
                "placebo_q95": pl.get("null_q95"), "placebo_pct": pl.get("pct"),
                "n_days_f1": dl.get("n_days_f1"), "n_days_base": dl.get("n_days_base"),
                "fail_reason": "",
            })
        except Exception as e:
            import traceback
            rec.update({"fail_reason": str(e)[:300]})
            print(f"[{a.tag}] {c['name']} ERROR {str(e)[:120]}", flush=True)
        pd.DataFrame([rec]).to_csv(a.out, mode="a", header=(started_empty and i == 0),
                                   index=False, encoding="utf-8-sig")
        done_n += 1
        print(f"[{a.tag} {done_n}/{len(mk)}] {c['name']} "
              f"d={rec.get('delta_mean')} p={rec.get('placebo_p')} {time.time()-t1:.0f}s", flush=True)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
