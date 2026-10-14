# -*- coding: utf-8 -*-
"""数据边界封装：强制 DATA_MAX（非标签）与 LABEL_MAX（标签型）。

所有 v2 流程的数据读取必须经过本模块。每次读取记录被读取的最大
signal_max_date 与 label_available_max_date，写入 JSONL 日志。
超过上界即抛错（§10.4、§10.5 的测试依赖此机制）。
"""
from __future__ import annotations

import json
import os

import pandas as pd

import _paths  # noqa: F401  注入 pandaai_online 以 import cache_utils
import windows
from cache_utils import find_stock_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOG = os.path.join(ROOT, "results", "merge_v2", "boundary_log.jsonl")
FACTORS_PATH = os.path.join(ROOT, "factor_lab_data", "factors.parquet")


def factor_data_version() -> str:
    """`factors.parquet` 的**声明式**数据版本（`factors_<md5[:8]>`）。

    供门控线 caliber 的 `data_version` 字段使用。**替代原先 3 处硬编码字面量**
    `"factors_abe94a59"`（2026-10-11）：值收拢到 `results/merge_v2/data_version.json`，
    当前即 `factors_abe94a59` ⇒ **零 caliber 失效**。

    **注意**：这是**声明式**（只在 `data_version.py --refresh` 时变），**不是**文件内容哈希。
    原因见 `windows.declared_data_version`：factors 有日频变体，若用内容哈希，
    门控 caliber 会**每日失效**导致全部重跑。
    """
    return windows.declared_data_version("factors", FACTORS_PATH)


def panel_data_version() -> str:
    """个股线面板的**声明式**数据版本（`panel_<md5[:8]>`），仅作**溯源记录**。

    **不参与个股线指纹过滤**（2026-10-11 用户裁定：指纹只用 `cut_date`）。
    见 `windows.declared_data_version`。
    """
    return windows.declared_data_version("panel", find_stock_features())


def _log(event: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_factors(columns, data_max: str, label: str = "") -> pd.DataFrame:
    """读取 factors.parquet 指定列，只返回信号日 ≤ data_max 的行。

    文件本身可能包含更新的数据（例如版本 V1 用 DATA_MAX=2024-06-21 读取
    到 2026-09-18 的文件）。此时只过滤，不报错；但记录文件最大日期与
    返回最大日期，供 §10.4 边界泄漏测试核对。返回为空则抛错。

    **缺列守卫（2026-10-12）**：数据代际变更会把缺列暴露成晦涩的
    `ArrowInvalid`（实测：外部于 2026-10-12 22:59 重生成 factors.parquet，
    262→155 列、起点 2023-06-01、缺 18 个 base20 因子）。本守卫**先列清单
    再拒绝**，并给出 provenance（行数/日期范围）与处置指引。
    """
    fp = os.path.join(ROOT, "factor_lab_data", "factors.parquet")
    want = [c for c in columns if c not in ("stock_code", "date")]
    import pyarrow.parquet as _pqp  # 仅取 schema，不读数据
    have = set(map(str, _pqp.read_schema(fp).names))
    missing = [c for c in want if c not in have]
    if missing:
        meta = _pqp.read_metadata(fp)
        raise SystemExit(
            f"read_factors 拒绝：factors.parquet 缺 {len(missing)}/{len(want)} 个请求列"
            f"（示例 {missing[:5]}…）。\n"
            f"  文件：{fp}\n"
            f"  provenance：行数 {meta.num_rows}；声明数据版本 {declared_version_note()}\n"
            f"  最可能原因：**数据代际变更**（上游重生成 factors.parquet，列裁剪/换代）。\n"
            f"  处置：① 确认本文件是否为正确世代（同目录 factors_free/normal_* 并存）；"
            f"② 若换代成立，先跑 data_version.py --refresh 并换登记窗口，"
            f"**再**重跑本批；不得静默换用其他列。")
    cols = ["stock_code", "date"] + want
    df = pd.read_parquet(fp, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    file_mx = df["date"].max()
    df = df[df["date"] <= pd.Timestamp(data_max)].copy()
    assert len(df), f"factors 在 DATA_MAX={data_max} 之前无数据"
    mx = df["date"].max()
    _log({"op": "read_factors", "label": label, "cols": len(cols),
          "rows": len(df), "signal_max_date": str(mx.date()), "data_max": data_max,
          "file_max_date": str(pd.Timestamp(file_mx).date())})
    return df


def declared_version_note() -> str:
    try:
        return f"{factor_data_version()}"
    except Exception:
        return "unknown"


def read_labels(columns, label_max: str, label: str = "") -> pd.DataFrame:
    """读取 labels_prod.parquet 指定列，只返回信号日 ≤ label_max 的行。语义同 read_factors。"""
    fp = os.path.join(ROOT, "results", "merge_v2", "labels_prod.parquet")
    cols = ["stock_code", "date"] + [c for c in columns if c not in ("stock_code", "date")]
    df = pd.read_parquet(fp, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    file_mx = df["date"].max()
    df = df[df["date"] <= pd.Timestamp(label_max)].copy()
    assert len(df), f"labels 在 LABEL_MAX={label_max} 之前无数据"
    mx = df["date"].max()
    _log({"op": "read_labels", "label": label, "cols": len(cols),
          "rows": len(df), "signal_max_date": str(mx.date()), "label_max": label_max,
          "file_max_date": str(pd.Timestamp(file_mx).date())})
    return df


def read_cache(columns, data_max: str, label: str = "") -> pd.DataFrame:
    """读取 cache 的 stock_features（自动发现，见 `cache_utils.find_stock_features`）。

    2026-10-11 修正：原为**硬编码** `stock_features_20200101_20260918.parquet`，
    而个股线 `evaluator.load_panel()` 用自动发现（规则=开始最早/期间最长）。
    两者在今天解析到同一文件，但缓存被管线重生成后（如出现同起始、更晚结束的文件）
    会自动切换而硬编码处不会 → **两条线静默分裂到不同面板**。改为同一发现逻辑以消除该隐患。
    """
    fp = find_stock_features()
    cols = ["stock_code", "date"] + [c for c in columns if c not in ("stock_code", "date")]
    df = pd.read_parquet(fp, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    file_mx = df["date"].max()
    df = df[df["date"] <= pd.Timestamp(data_max)].copy()
    assert len(df), f"cache 在 DATA_MAX={data_max} 之前无数据"
    mx = df["date"].max()
    _log({"op": "read_cache", "label": label, "cols": len(cols),
          "rows": len(df), "signal_max_date": str(mx.date()), "data_max": data_max,
          "file_max_date": str(pd.Timestamp(file_mx).date())})
    return df
