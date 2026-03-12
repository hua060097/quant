import json
import numpy as np

path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())
recent = dates[-60:]  # 最近20天，你也可以改成[-60:]

cond_gap = []
uncond_gap = []
score_margin = []
recall = []
serial_ret = []

for d in recent:
    rec = data[d]

    # gaps
    if rec.get("conditional_gap") is not None:
        cond_gap.append(rec["conditional_gap"])
    if rec.get("unconditional_gap") is not None:
        uncond_gap.append(rec["unconditional_gap"])

    # score margin
    if rec.get("conditional_score_margin_top1_top2") is not None:
        score_margin.append(rec["conditional_score_margin_top1_top2"])

    # serial top1 realized return
    st1 = rec.get("serial_top1") or {}
    if st1.get("real_return") is not None:
        serial_ret.append(st1["real_return"])

    # recall: is rank_all_top1 in cls_topn?
    ra = rec.get("rank_all_top1") or {}
    ra_code = ra.get("code")
    cls_list = rec.get("cls_topn") or []
    cls_codes = {x.get("code") for x in cls_list if x.get("code")}

    if ra_code is not None and len(cls_codes) > 0:
        recall.append(1.0 if ra_code in cls_codes else 0.0)

print("recent20 stats:")
print("conditional_gap_mean =", float(np.mean(cond_gap)) if cond_gap else None)
print("unconditional_gap_mean =", float(np.mean(uncond_gap)) if uncond_gap else None)
print("conditional_score_margin_mean =", float(np.mean(score_margin)) if score_margin else None)
print("recall_rankall_top1_in_cls_mean =", float(np.mean(recall)) if recall else None)
print("serial_top1_mean_return =", float(np.mean(serial_ret)) if serial_ret else None)