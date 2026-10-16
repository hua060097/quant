# -*- coding: utf-8 -*-
"""门控交互配对指纹缓存（f1_batch / f1_sector 共享）：pair 级去重续算。

pair_fp = sha12(base|gate|s|d|form|label_max|data_version)：同一口径配对在
同一数据版本下只算一次（200 次置换是主要成本）。缓存文件
results/merge_v2/gates_search/f1_pair_cache.jsonl（append-only JSONL，
每行 {fp, result}；result 含全部统计量 + fail_reason + 写入时间）。

命中（同 fp 且无 fail_reason）：调用方直接复用统计量，0 秒跳过置换。
--out 改为全量去重覆盖写（按 name  keep-last），多次运行幂等。
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
SPACE = os.path.join(ROOT, "results", "merge_v2", "gates_search")
CACHE_PATH = os.path.join(SPACE, "f1_pair_cache.jsonl")

#: 复用时直接透传进 rec 的统计量键（fail_reason 单独处理）。
STAT_KEYS = ("delta_mean", "delta_ci_lo", "delta_ci_hi", "nblocks_pos", "blocks",
             "slope_b", "slope_t", "placebo_p", "placebo_obs", "placebo_q95",
             "placebo_pct", "n_days_f1", "n_days_base", "tilt_delta", "delta_net")


def pair_fp(base: str, gate: str, s, d, form: str, label_max: str,
            data_version: str = "") -> str:
    """配对唯一身份（含口径与数据版本；任一变化即新身份）。"""
    raw = "|".join([str(base), str(gate), str(s), str(d), str(form),
                    str(label_max), str(data_version or "")])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_cache(path: str = CACHE_PATH) -> dict:
    """读缓存 → {fp: result}（同 fp keep-last）。文件缺失返回空。"""
    out: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("fp"):
                    out[str(r["fp"])] = r.get("result") or {}
    except FileNotFoundError:
        pass
    return out


def save_result(fp: str, result: dict, path: str = CACHE_PATH) -> None:
    """追加一条结果（含 fail_reason 的也存，供调用方区分跳过/复算）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"fp": str(fp), "result": result},
                           ensure_ascii=False) + "\n")


def cache_hit(fp: str, cache: dict) -> dict | None:
    """可复用命中：有记录、无 fail_reason、且含核心统计量。否则 None（需重算）。"""
    r = cache.get(str(fp))
    if not r or r.get("fail_reason"):
        return None
    if r.get("delta_mean") is None and r.get("placebo_p") is None:
        return None
    return r
