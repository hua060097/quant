from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


_STORE_DIR = Path(__file__).resolve().parent / "frozen_stats"
_STORE_DIR.mkdir(parents=True, exist_ok=True)
_LOGGER = logging.getLogger(__name__)


def _safe_str(value: Any) -> str:
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")


def _store_file(kind: str, module_name: str, factor_name: str, train_start: Any, train_end: Any, extra: str) -> Path:
    base = "__".join(
        [
            kind,
            _safe_str(module_name),
            _safe_str(factor_name),
            _safe_str(train_start),
            _safe_str(train_end),
            _safe_str(extra),
        ]
    )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return _STORE_DIR / f"{base[:160]}__{digest}.json"


def _source_version_tag(module_file: str | None) -> str:
    if not module_file:
        return "unknown"
    try:
        payload = Path(module_file).read_bytes()
        return hashlib.sha1(payload).hexdigest()[:16]
    except Exception:
        return "unknown"


def load_frozen_clipper_params(
    *,
    module_name: str,
    factor_name: str,
    train_start: Any,
    train_end: Any,
    n_mad: float,
    module_file: str | None = None,
) -> dict[str, Any] | None:
    path = _store_file("clipper", module_name, factor_name, train_start, train_end, f"nmad_{n_mad}")
    if not path.exists():
        _LOGGER.info(
            "[FrozenStore] MISS clipper %s/%s train=%s~%s n_mad=%s reason=file_not_found",
            module_name,
            factor_name,
            train_start,
            train_end,
            n_mad,
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOGGER.warning(
            "[FrozenStore] MISS clipper %s/%s train=%s~%s n_mad=%s reason=read_error error=%s",
            module_name,
            factor_name,
            train_start,
            train_end,
            n_mad,
            exc,
        )
        return None
    version_tag = _source_version_tag(module_file)
    if payload.get("version_tag") != version_tag:
        _LOGGER.info(
            "[FrozenStore] MISS clipper %s/%s train=%s~%s n_mad=%s reason=version_mismatch",
            module_name,
            factor_name,
            train_start,
            train_end,
            n_mad,
        )
        return None
    _LOGGER.info(
        "[FrozenStore] HIT clipper %s/%s train=%s~%s n_mad=%s file=%s",
        module_name,
        factor_name,
        train_start,
        train_end,
        n_mad,
        path.name,
    )
    return payload


def save_frozen_clipper_params(
    *,
    module_name: str,
    factor_name: str,
    train_start: Any,
    train_end: Any,
    n_mad: float,
    module_file: str | None,
    median: float,
    mad: float,
    lower: float,
    upper: float,
    fit_sample_count: int,
) -> None:
    path = _store_file("clipper", module_name, factor_name, train_start, train_end, f"nmad_{n_mad}")
    payload = {
        "kind": "clipper",
        "module_name": module_name,
        "factor_name": factor_name,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "n_mad": float(n_mad),
        "median": float(median),
        "mad": float(mad),
        "lower": float(lower),
        "upper": float(upper),
        "fit_sample_count": int(fit_sample_count),
        "version_tag": _source_version_tag(module_file),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info(
        "[FrozenStore] SAVE clipper %s/%s train=%s~%s n_mad=%s samples=%s file=%s",
        module_name,
        factor_name,
        train_start,
        train_end,
        n_mad,
        fit_sample_count,
        path.name,
    )


def load_frozen_scalar_params(
    *,
    module_name: str,
    stat_name: str,
    train_start: Any,
    train_end: Any,
    module_file: str | None = None,
) -> dict[str, Any] | None:
    path = _store_file("scalar", module_name, stat_name, train_start, train_end, "scalar")
    if not path.exists():
        _LOGGER.info(
            "[FrozenStore] MISS scalar %s/%s train=%s~%s reason=file_not_found",
            module_name,
            stat_name,
            train_start,
            train_end,
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOGGER.warning(
            "[FrozenStore] MISS scalar %s/%s train=%s~%s reason=read_error error=%s",
            module_name,
            stat_name,
            train_start,
            train_end,
            exc,
        )
        return None
    version_tag = _source_version_tag(module_file)
    if payload.get("version_tag") != version_tag:
        _LOGGER.info(
            "[FrozenStore] MISS scalar %s/%s train=%s~%s reason=version_mismatch",
            module_name,
            stat_name,
            train_start,
            train_end,
        )
        return None
    _LOGGER.info(
        "[FrozenStore] HIT scalar %s/%s train=%s~%s file=%s",
        module_name,
        stat_name,
        train_start,
        train_end,
        path.name,
    )
    return payload


def save_frozen_scalar_params(
    *,
    module_name: str,
    stat_name: str,
    train_start: Any,
    train_end: Any,
    module_file: str | None,
    value: float,
    fit_sample_count: int,
) -> None:
    path = _store_file("scalar", module_name, stat_name, train_start, train_end, "scalar")
    payload = {
        "kind": "scalar",
        "module_name": module_name,
        "stat_name": stat_name,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "value": float(value),
        "fit_sample_count": int(fit_sample_count),
        "version_tag": _source_version_tag(module_file),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info(
        "[FrozenStore] SAVE scalar %s/%s train=%s~%s samples=%s file=%s",
        module_name,
        stat_name,
        train_start,
        train_end,
        fit_sample_count,
        path.name,
    )
