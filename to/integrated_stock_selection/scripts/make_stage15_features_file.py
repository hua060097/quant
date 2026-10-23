#!/usr/bin/env python3
"""Convert a merge features_manifest.json into stage15 --features_file JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    if isinstance(src, dict):
        feats = src.get('features_all') or src.get('features') or src.get('factors')
    else:
        feats = src
    if not isinstance(feats, list) or not feats:
        raise SystemExit(f'no feature list in {a.manifest}')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'features': [str(x) for x in feats]},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{a.manifest} -> {out} ({len(feats)} features)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
