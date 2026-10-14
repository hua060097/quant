# -*- coding: utf-8 -*-
"""把个股台账里**缺 `data_cut` 的旧记录显式标记**为历史遗留截止日。

背景（2026-10-11 用户指令）：候选身份 = 公式 × 数据截止日。个股线历史上
（`fitness.py` 的 `END=2026-08-21` 期间）用含 2026 年数据的窗口做过搜索，
其指纹需要被**标记**而不参与后续以默认截止日（`windows.DEFAULT_DATA_CUT`,
2025-12-31）进行的搜索/筛选。

本脚本只做**标记**：
  - 缺 `data_cut` 的记录 → `"data_cut": windows.LEGACY_DATA_CUT`,
    `"data_cut_legacy": true`（后者表示该截止日系事后推断，非写入时记录）
  - 已有 `data_cut` 的记录**不动**

安全性：
  - 默认 **dry-run**（只报告，不写盘）；`--apply` 才实际写入
  - 写入前备份 `ledger_backup_pre_cut_<timestamp>.jsonl`
  - 写临时文件 → 校验（行数一致、逐行可解析、除新增字段外内容一致）→ 原子替换
  - **删除 sidecar 摘要**（`ledger_summary.json`）：台账内容变了，摘要必须全量重建，
    否则按 byte_offset 增量消费会读到行中间而静默丢失记录

用法：
    python migrate_data_cut.py                 # 预览
    python migrate_data_cut.py --apply         # 执行
    python migrate_data_cut.py --apply --cut 2026-08-21   # 指定遗留截止日
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401
import windows

HERE = Path(__file__).resolve().parent
DEFAULT_LEDGER = HERE / 'ledger.jsonl'


def _load(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]


def _content_equal(a: dict, b: dict, ignore: set[str]) -> bool:
    ka = {k: v for k, v in a.items() if k not in ignore}
    kb = {k: v for k, v in b.items() if k not in ignore}
    return ka == kb


def migrate(path: Path, cut: str, apply: bool) -> dict:
    if not path.exists():
        raise SystemExit(f'台账不存在：{path}')
    lines = _load(path)
    out_lines: list[str] = []
    n_marked = n_have = n_bad = 0
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            n_bad += 1
            out_lines.append(ln)
            continue
        if not isinstance(rec, dict):
            n_bad += 1
            out_lines.append(ln)
            continue
        if rec.get('data_cut'):
            n_have += 1
            out_lines.append(ln)
            continue
        marked = dict(rec)
        marked['data_cut'] = cut
        marked['data_cut_legacy'] = True
        out_lines.append(json.dumps(marked, ensure_ascii=False))
        n_marked += 1

    result = {'path': str(path), 'n_lines': len(lines), 'n_marked': n_marked,
              'n_already_had': n_have, 'n_unparsable': n_bad, 'cut': cut,
              'applied': False}
    if not apply:
        return result

    newest = path.with_name(f'ledger_backup_pre_cut_'
                            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
    shutil.copy2(path, newest)

    tmp = path.with_suffix('.jsonl.tmp')
    tmp.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')

    # 校验：行数一致、逐行可解析、除新增字段外内容一致
    chk = [x for x in tmp.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(chk) != len(lines):
        tmp.unlink()
        raise SystemExit(f'校验失败：行数 {len(chk)} != {len(lines)}')
    for i, (a, b) in enumerate(zip(lines, chk)):
        da, db = json.loads(a), json.loads(b)
        if not _content_equal(da, db, {'data_cut', 'data_cut_legacy'}):
            tmp.unlink()
            raise SystemExit(f'校验失败：第 {i + 1} 行内容变化超出预期')

    tmp.replace(path)

    side = path.with_name(path.stem + '_summary.json')
    if side.exists():
        side.unlink()
        result['sidecar_removed'] = str(side)

    result['applied'] = True
    result['backup'] = str(newest)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ledger', default=str(DEFAULT_LEDGER))
    ap.add_argument('--cut', default=windows.LEGACY_DATA_CUT,
                    help='要打的遗留截止日（默认 %s）' % windows.LEGACY_DATA_CUT)
    ap.add_argument('--apply', action='store_true', default=False,
                    help='实际写入（默认只预览）')
    args = ap.parse_args()

    cut = windows.normalize_data_cut(args.cut)
    res = migrate(Path(args.ledger), cut, args.apply)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not args.apply:
        print('\n（预览模式；加 --apply 才会写入）', file=sys.stderr)


if __name__ == '__main__':
    main()
