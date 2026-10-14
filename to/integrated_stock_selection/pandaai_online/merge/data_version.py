# -*- coding: utf-8 -*-
"""数据版本声明工具（**显式**换代杠杆）。

背景（2026-10-11 用户裁定）：
- `cache/stock_features_*` 是**日频重生成**的（文件名带日期、内容每日追加）；
  `factors` 也有日频变体。若把文件内容哈希当作**指纹作用键**，则每个交易日
  全部指纹失效，「已测过，跳过」机制失去意义。
- 但数据也可能发生**实质变化**（批量复权重算、数据源更换、修数据），此时
  旧结论不再可比，**应当**触发补测。

因此本工具把两种情形分开：
- **日常更新**：什么都不用做。指纹只用 `--data-cut`，不会失效。
- **实质变化**：运行 `python data_version.py --refresh --kind factors --note "..."`，
  声明新版本 → 门控 caliber 变化 → 自动补测。
  （个股线指纹**不吃**该声明；如需重测个股线，用 `run_search --retest-before <日期>`。）

用法：
    python data_version.py                        # 查看当前声明
    python data_version.py --refresh --kind factors --note "2026-11 批量复权重算"
    python data_version.py --refresh --kind panel
    python data_version.py --inspect              # 附：显示文件内容哈希（不写声明）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import _paths  # noqa: F401
import boundary as B
import windows as W


def _source(kind: str) -> str:
    if kind == "factors":
        return B.FACTORS_PATH
    if kind == "panel":
        return str(B.find_stock_features())
    raise SystemExit(f"未知 kind={kind!r}（可选 factors / panel）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true",
                    help="按当前文件内容哈希**重新声明**该 kind 的版本（显式换代）")
    ap.add_argument("--kind", default=None, choices=("factors", "panel"),
                    help="要刷新的数据类型（--refresh 时必填）")
    ap.add_argument("--note", default="", help="刷新原因（写入声明文件，供审计）")
    ap.add_argument("--inspect", action="store_true",
                    help="只显示当前文件的内容哈希（不写任何声明）")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fp = W.DECLARED_VERSION_FILE
    cur: dict = {}
    if fp.exists():
        try:
            cur = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            cur = {}

    print("=" * 74)
    print("数据版本声明（%s）" % ("**只由本工具显式更新**，不随日常 cache 更新变化"))
    print("=" * 74)
    print("  声明文件: %s" % fp)
    print("  存在    : %s" % fp.exists())
    for kind in ("factors", "panel"):
        v = cur.get(kind)
        print("  %-8s: %s" % (kind, v or "(未声明)"))
        if cur.get(f"{kind}_refreshed_at"):
            print("            refreshed_at = %s" % cur[f"{kind}_refreshed_at"])
        if cur.get(f"{kind}_note"):
            print("            note         = %s" % cur[f"{kind}_note"])

    if a.inspect:
        print()
        print("--- 文件内容哈希（仅供对照；**不是**指纹作用键）---")
        for kind in ("factors", "panel"):
            try:
                print("  %-8s: %s" % (kind, W.file_data_version(_source(kind), kind)))
            except Exception as exc:
                print("  %-8s: 失败 %s" % (kind, exc))
        return 0

    if a.refresh:
        if not a.kind:
            raise SystemExit("--refresh 需要 --kind factors|panel")
        src = _source(a.kind)
        if not os.path.isfile(src):
            raise SystemExit("源文件不存在: %s" % src)
        res = W.refresh_declared_version(a.kind, src, note=a.note)
        print()
        print("刷新结果:")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        if res["changed"]:
            print()
            print("⚠ 版本已改变 ⇒ 依赖该版本的口径（门控 caliber）会判为 stale 并补测。")
            if a.kind == "factors":
                print("  门控线：`run_gate_pipeline.py` 会按新 caliber 重跑。")
                print("  `confirm_candidate` 的 F1 台账记录也会因 caliber 变化而失效。")
        else:
            print()
            print("版本未变 ⇒ 无任何口径失效（内容与当前声明一致）。")
        return 0

    print()
    print("日常更新：什么都不用做（指纹只用 --data-cut）。")
    print("实质变化：--refresh --kind <factors|panel> --note \"原因\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
