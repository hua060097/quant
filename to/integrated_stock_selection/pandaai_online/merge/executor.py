"""远程批次导出（零平台调用）：生成 batch.py 的 candidates.txt 与 meta.json。

- 格式：每行 `name ~ formula ~ direction`，`#` 注释；用 skill 自带 batch.py 的解析器做校验。
- 只写文件，不创建/不运行任何平台对象；credit 预估仅作提示。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401

SKILL_BATCH = (Path.home() / '.claude' / 'skills' / 'pandaai-factor-online'
               / 'scripts' / 'batch.py')


def _load_batch_module():
    if not SKILL_BATCH.exists():
        return None
    spec = importlib.util.spec_from_file_location('skill_batch', SKILL_BATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_file(path: Path) -> tuple[bool, str]:
    module = _load_batch_module()
    if module is None:
        return False, f'skill batch.py not found: {SKILL_BATCH}'
    try:
        candidates = module.parse_candidates(Path(path), mode='formula')
    except SystemExit as exc:
        return False, f'parse failed: {exc}'
    return True, f'{len(candidates)} candidates parsed and linted'


def write_candidates(rows: list[dict], out_path: Path, tag: str = '',
                     notes: tuple = (), meta_path: Path | None = None) -> dict:
    """rows: [{name, formula, direction, expression_hash?, family?, sia?, turnover?, ...}]"""
    names = [r['name'] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('duplicate candidate names')
    for r in rows:
        if '~' in r['name'] or '~' in r['formula']:
            raise ValueError(f"candidate {r['name']} contains '~'")
        if str(r['direction']) not in ('0', '1'):
            raise ValueError(f"candidate {r['name']} direction must be 0/1")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'# {tag} | generated {datetime.now().isoformat(timespec="seconds")} '
             f'| {len(rows)} candidates | zero platform calls (export only)']
    lines.extend(f'# {n}' for n in notes)
    for r in rows:
        lines.append(f"{r['name']} ~ {r['formula']} ~ {int(r['direction'])}")
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    meta = {
        'tag': tag,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'file': out_path.name,
        'candidates': [{k: r.get(k) for k in
                        ('name', 'direction', 'expression_hash', 'family', 'sia',
                         'turnover', 'net', 'rank_ic', 'ic_ir', 'win', 'pool_corr_max')}
                       for r in rows],
    }
    meta_path = Path(meta_path) if meta_path else out_path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'txt': str(out_path), 'meta': str(meta_path), 'n': len(rows)}


def estimate_credits(n_candidates: int, per_run: int = 2) -> dict:
    probe = n_candidates * per_run
    full = max(0, round(n_candidates / 3)) * per_run
    return {'probe_credits': probe, 'full_estimate': full,
            'total_estimate': probe + full}


def main() -> None:
    ap = argparse.ArgumentParser(description='校验 candidates.txt 是否符合 batch.py 格式')
    ap.add_argument('file', type=Path)
    args = ap.parse_args()
    ok, msg = validate_file(args.file)
    print(('OK  ' if ok else 'FAIL') + ' ' + msg)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
