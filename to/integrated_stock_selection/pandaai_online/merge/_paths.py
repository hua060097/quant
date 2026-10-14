"""sys.path 注入：merge 目录（pandaai_online/merge）+ pandaai_online 目录。"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (str(_HERE), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
