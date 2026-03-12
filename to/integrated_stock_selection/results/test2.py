import json
import numpy as np

path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"

with open(path,"r",encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

top1_ret=[]
top2_ret=[]
top1_top2_gap=[]

for d in dates:

    rec=data[d]

    serial=rec.get("serial_top1")
    cls=rec.get("cls_topn")

    if not serial or not cls:
        continue

    # Top1
    r1=serial["real_return"]

    # 找候选域里排序第2
    sorted_cls=sorted(cls,key=lambda x:x["rank_score"],reverse=True)

    if len(sorted_cls)<2:
        continue

    r2=sorted_cls[1]["real_return"]

    top1_ret.append(r1)
    top2_ret.append(r2)
    top1_top2_gap.append(r1-r2)


print("Top1 mean:",np.mean(top1_ret))
print("Top2 mean:",np.mean(top2_ret))
print("Top1-Top2 mean:",np.mean(top1_top2_gap))

print("Top1>Top2 ratio:",np.mean(np.array(top1_top2_gap)>0))