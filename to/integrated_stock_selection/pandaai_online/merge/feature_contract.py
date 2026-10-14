# -*- coding: utf-8 -*-
"""FeatureContract：特征定义 / 数据集 / 实验三层哈希 + 时间字段（批次 4.3）。

**为什么**（Codex P1 采纳，2026-10-12）：因子筛选与训练系统之间需要一个
明确、可验证、可追溯的接口。**不要一个总哈希**——日期范围变化应触发新的
数据集/实验身份，而不必然改变特征定义；哈希不一致**禁止静默复用旧产物**，
但允许显式创建新的可追溯实验版本（"契约的目的不是禁止变化，而是让变化
不能悄悄发生"）。

三层（2026-10-13 起 **四层**）哈希（各管边界，禁止重复计数）：

| 哈希 | 绑定 | 不含 |
|---|---|---|
| `feature_definition_hash` | **特征定义**：候选标签因子集（base20）、族清单引用、池状态 | 标签定义、样本范围、折切分、超参 |
| `label_definition_hash` | **标签定义**：公式/阈值/未来区间/回撤口径（2026-10-13 从特征哈希中**拆出独立**——修正"标签变了但特征没变"无法单独追踪） | 特征定义、数据、超参 |
| `dataset_hash` | 样本范围（数据文件 md5/行数/日期范围）+ 数据版本声明 + 股票池与缺失值规则 | 特征定义、超参 |
| `experiment_hash` | 特征池版本 + `label_definition_hash` + 折切分 + 训练参数 + 评价规则（=前三种 + 实验配置） | — |

冻结声明 = `results/merge_v2/contract.json`（data_version / label_def / folds /
hp_grid / dates）。**时间字段**（四层，按 §8.14/contract dates）：
`signal_max` / `feature_available_max` / `label_coverage`（r3/r5 的 T+k 覆盖区间）/
`label_available_max`。

用法：
    python feature_contract.py --show        # 打印三层哈希与四时间字段
    python feature_contract.py --out FILE    # 落盘 JSON（默认 contract_hashes.json）
    python feature_contract.py --check FILE  # 校验落盘与当前一致（不一致 exit=2）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
CONTRACT = os.path.join(RESULTS, "contract.json")
OUT_DEFAULT = os.path.join(RESULTS, "contract_hashes.json")

#: 训练特征池（Baseline = 生产池成员）。池冻结文件尚无 → 显式声明"baseline 未冻结"，
#: feature_definition_hash 现阶段绑定 **base20（候选标签因子集）+ 登记册定义**，
#: 并在哈希载荷中记录 status=baseline_pool_not_frozen（以免暗含池已冻结）。
#: S8_Stock_Pool 执行 POOL_S3/冻结后，此处换成 `--features pool_card.json`。
DEFAULT_DEFINITIONS = os.path.join(RESULTS, "base20.json")


def _sha12(canonical: str) -> str:
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def load_contract() -> dict:
    with open(CONTRACT, encoding="utf-8") as f:
        return json.load(f)


def definition_payload(extra_files: list[str] | None = None) -> dict:
    """特征定义载荷：**只含特征侧**（候选标签因子集定义 + 族清单引用 + 池状态）。

    2026-10-13 修正（Codex）：此前把标签定义混进本哈希 → "标签变了但特征没变"
    无法单独追踪。现拆出 `label_definition_hash`（见 label_payload）。
    """
    b20 = json.load(open(DEFAULT_DEFINITIONS, encoding="utf-8"))["base20"]
    defs = {
        "base20": [{"factor": x["factor"], "direction": x["direction"]} for x in b20],
        # 登记册当前世代的候选定义（族成员 = 待验证交互的定义）
        "family_manifest": "gates_search/family_manifest_F1.json",
        "baseline_pool_status": "baseline_pool_not_frozen（S8_Stock_Pool 冻结后替换）",
    }
    for i, fp in enumerate(extra_files or []):
        with open(fp, encoding="utf-8") as f:
            defs[f"extra_{i}:{os.path.basename(fp)}"] = json.load(f)
    return defs


def label_payload() -> dict:
    """标签定义载荷（**独立哈希**——与特征定义分离，2026-10-13 修正）。"""
    c = load_contract()
    return {"label_family": c["label_family"], "label_def": c["label_def"]}


def label_definition_hash() -> str:
    canon = json.dumps(label_payload(), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"))
    return _sha12(canon)


def feature_definition_hash(extra_files: list[str] | None = None) -> tuple[str, dict]:
    payload = definition_payload(extra_files)
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _sha12(canon), payload


def dataset_payload() -> dict:
    """样本范围 + 数据版本（md5 实测）+ 缺失值/复权口径声明。"""
    import pyarrow.parquet as pq
    fp = os.path.join(ROOT, "factor_lab_data", "factors.parquet")
    sch = pq.read_schema(fp).names
    meta = pq.read_metadata(fp)
    d = pd_date_bounds(fp)
    return {
        "factors_file": os.path.relpath(fp, ROOT).replace("\\", "/"),
        "factors_md5": factor_md5(),
        "factors_rows": meta.num_rows,
        "factors_cols": len(sch),
        "factors_dates": f"{d[0]}~{d[1]}",
        "cache": cache_note(),
        "labels_file": r"results/merge_v2/labels_prod.parquet",
        "missing_policy": ("缺失值规则随特征定义文本冻结（§8.1 逐条声明）；"
                           "本契约只记录口径级别：NaN 不填充，分位数用非 NaN 日"),
        "adjusted_price": "daily fq（cache 原口径，见 evaluator.load_panel）",
        "declared_data_version": B.factor_data_version(),
    }


def factor_md5() -> str:
    import hashlib
    h = hashlib.md5()
    fp = os.path.join(ROOT, "factor_lab_data", "factors.parquet")
    with open(fp, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_note() -> str:
    from cache_utils import find_stock_features
    p = str(find_stock_features())
    mtime = os.path.getmtime(p)
    import datetime
    return (os.path.basename(p) + " (mtime "
            + datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d") + ")")


def pd_date_bounds(fp: str) -> tuple[str, str]:
    import pandas as pd
    d = pd.read_parquet(fp, columns=["date"])["date"]
    return str(pd.to_datetime(d).min().date()), str(pd.to_datetime(d).max().date())


def experiment_payload(feat_hash: str, ds_hash: str, label_hash: str) -> dict:
    """实验配置 = 特征池版本 + 标签版本（**哈希**）+ 引擎声明 + 评价规则。"""
    c = load_contract()
    return {
        "feature_definition_hash": feat_hash,
        "dataset_hash": ds_hash,
        "label_definition_hash": label_hash,
        "pool_version": "baseline_pool_not_frozen（pending S8_Stock_Pool 冻结）",
        "label_version": c["label_family"],
        "folds": c["folds"],
        "engine": {
            "script": "main_alpha_stage15_hp_search.py（仓库根；merge 不内嵌模型级增量诊断，"
                      "用户裁定 2026-10-13）",
            "ab_protocol": ("正式对照 = 两次生产运行（--features_file：基线特征集 / 基线+候选）"
                            "+ compare_stage15.py 对拍；折/种子/预算/交易核算由生产脚本保证"),
        },
        "eval_note": ("验收关注 Top1 类指标与真实可实现性（生产引擎含槽位净值/成本折算）；"
                      "merge 侧五项预规定指标描述废止（随 s8_test_b.py 删除，§16.29.1）"),
        "dates": c["dates"],
    }


def compute(extra_definition_files: list[str] | None = None) -> dict:
    fh, _ = feature_definition_hash(extra_definition_files)
    lh = label_definition_hash()
    dp = dataset_payload()
    canon_ds = json.dumps(dp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    dh = _sha12(canon_ds)
    ep = experiment_payload(fh, dh, lh)
    canon_ep = json.dumps(ep, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    eh = _sha12(canon_ep)
    c = load_contract()
    return {
        "feature_definition_hash": fh,
        "label_definition_hash": lh,
        "dataset_hash": dh,
        "experiment_hash": eh,
        "time_fields": {
            "signal_max": c["dates"]["signal_max_production"],
            "feature_available_max": c["dates"]["signal_max_production"],
            "label_coverage": ("r3=T+2..T+3（权重 0.7/0.3）；r5=T+4..T+5；"
                               "dd3=min(T+2,T+3)，T+1 排除"),
            "label_available_max": c["dates"]["label_available_max_production"],
            "forward_first_signal": c["dates"]["forward_first_signal_date"],
        },
        "dataset_payload": dp,
        "experiment_payload": ep,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None, help="落盘 JSON（默认 contract_hashes.json）")
    ap.add_argument("--check", default=None, help="校验已落盘与当前一致")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cur = compute()
    if a.check:
        with open(a.check, encoding="utf-8") as f:
            old = json.load(f)
        keys = ("feature_definition_hash", "label_definition_hash", "dataset_hash",
                "experiment_hash")
        diffs = [k for k in keys if old.get(k) != cur[k]]
        print(json.dumps({"ok": not diffs, "diff_keys": diffs,
                          "old": {k: old.get(k) for k in keys},
                          "new": {k: cur[k] for k in keys}},
                         ensure_ascii=False, indent=1))
        return 0 if not diffs else 2
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=1)
    if a.show or a.out:
        print(f"feature_definition_hash = {cur['feature_definition_hash']}")
        print(f"label_definition_hash   = {cur['label_definition_hash']}（独立——2026-10-13 拆分）")
        print(f"dataset_hash            = {cur['dataset_hash']}")
        print(f"experiment_hash         = {cur['experiment_hash']}")
        tf = cur["time_fields"]
        print(f"time_fields: signal_max={tf['signal_max']} "
              f"label_available_max={tf['label_available_max']} "
              f"({tf['label_coverage']})")
        if a.out:
            print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
