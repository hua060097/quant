# -*- coding: utf-8 -*-
"""自定义因子（论文 / Alpha101 / Python 向量化函数）统一注册与求值中心（V4.1 Stage2）。

支持两种注册方式：
  1. DSL 公式型：register_formula_factor(factor_id, formula, axis, description)
  2. Python 函数型：@register_py_factor(factor_id, axis, description)
     函数签名：fn(frame: pd.DataFrame, ev: Evaluator) -> pd.Series | np.ndarray
     传入的 frame 已按 (stock_code, date) 排序且已调用 ensure_local_fields(frame)，
     可直接访问 OPEN/HIGH/LOW/CLOSE/VWAP/VOLUME/AMOUNT/TURNOVER/MARKET_CAP/
     MKT_RET/IND_RET/IND_EXCESS/DISP/BETA20/BETA60/IVOL20 等全部列。

安全约束：Python函数禁`shift(-n)`前视（由调用方`audit_cut_paths`抽查）；
返回必须与frame等长，有限值外一律NaN；`competition_uploadable`仅公式型经
`executor.competition_status`判定，Python型一律False（只走本地VWAP）。
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from evaluator import Evaluator, ensure_extended_fields, ensure_local_fields
from parser import parse_formula


@dataclass
class CustomFactorSpec:
    factor_id: str
    axis: str
    family: str
    kind: str  # 'formula' | 'python'
    formula_str: str
    fn: Callable | None = None
    description: str = ''
    competition_uploadable: bool = False

    @property
    def spec_hash(self) -> str:
        if self.kind == 'formula':
            payload = f"formula|{self.factor_id}|{self.formula_str.strip()}"
        else:
            src = inspect.getsource(self.fn) if self.fn is not None else self.factor_id
            payload = f"python|{self.factor_id}|{src}"
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


REGISTRY: dict[str, CustomFactorSpec] = {}


def register_formula_factor(factor_id: str, formula: str, axis: str = 'custom',
                            description: str = '') -> CustomFactorSpec:
    from executor import competition_status
    ok_comp, _ = competition_status(formula)
    spec = CustomFactorSpec(
        factor_id=factor_id, axis=axis, family=f'custom_{axis}',
        kind='formula', formula_str=formula, fn=None,
        description=description, competition_uploadable=ok_comp
    )
    REGISTRY[factor_id] = spec
    return spec


def register_py_factor(factor_id: str, axis: str = 'custom', description: str = ''):
    def _decorator(fn: Callable) -> Callable:
        spec = CustomFactorSpec(
            factor_id=factor_id, axis=axis, family=f'custom_{axis}',
            kind='python', formula_str=f"py:{fn.__name__}", fn=fn,
            description=description, competition_uploadable=False
        )
        REGISTRY[factor_id] = spec
        return fn
    return _decorator


def _evaluate_core(spec: CustomFactorSpec, frame: pd.DataFrame,
                   ev: Evaluator | None = None) -> np.ndarray:
    """求值内核（audit_pit内部复用，不触发审计，避免递归）。"""
    ensure_extended_fields(frame)
    ev = ev or Evaluator(frame)
    if spec.kind == 'formula':
        node = parse_formula(spec.formula_str)
        vals = ev.values(node)
    else:
        assert spec.fn is not None
        res = spec.fn(frame, ev)
        vals = res.to_numpy(dtype=float) if isinstance(res, pd.Series) else np.asarray(res, dtype=float)
    if len(vals) != len(frame):
        raise ValueError(f"Custom factor {spec.factor_id} returned length {len(vals)} != frame {len(frame)}")
    return np.where(np.isfinite(vals), vals, np.nan)


def evaluate_custom_factor(spec: CustomFactorSpec, frame: pd.DataFrame,
                           ev: Evaluator | None = None) -> np.ndarray:
    """在给定面板上求值自定义因子，返回与 frame 行对齐的 float64 数组。

    每个spec_hash每进程审计一次PIT（warn-only，不阻断求值）。
    """
    key = spec.spec_hash
    if key not in _PIT_AUDITED:
        _PIT_AUDITED.add(key)
        try:
            # 大面板采样200只做审计（PIT无前视性质与样本量无关），防3次全量拷贝爆内存
            aframe = frame
            _codes = pd.Series(frame["stock_code"].unique())
            if len(frame) > 200000 and len(_codes) > 200:
                _pick = set(_codes.sample(200, random_state=7).astype(str))
                aframe = frame[frame["stock_code"].astype(str).isin(_pick)].copy()
            rep = audit_pit(spec, aframe)
            if not rep.get("passed"):
                print(f"warning: custom factor {spec.factor_id} PIT审计未通过 "
                      f"{rep.get('failed_dates')}（仅告警，不阻断）", flush=True)
        except Exception as e:
            print(f"warning: custom factor {spec.factor_id} PIT审计异常 "
                  f"{str(e)[:120]}（仅告警）", flush=True)
    return _evaluate_core(spec, frame, ev)


_PIT_AUDITED: set[str] = set()


def audit_pit(spec: CustomFactorSpec, frame: pd.DataFrame,
              n_probes: int = 3, seed: int = 7) -> dict:
    """时间穿越扰动PIT审计（PLAN §10.3思想，warn-only不阻断）。

    随机取n_probes个日期T：把T+1之后全部行情列改成随机数后重算，
    断言≤T的因子值完全不变（np.allclose，equal_nan=True）。
    返回 {passed, failed_dates, note}；调用方只告警不拦截。
    """
    rng = np.random.default_rng(seed)
    dates = np.sort(pd.to_datetime(frame["date"]).unique())
    if len(dates) < 5:
        return {"passed": True, "failed_dates": [], "note": "too few dates, skipped"}
    cands = rng.choice(np.arange(1, len(dates) - 1),
                       size=min(n_probes, len(dates) - 2), replace=False)
    base = _evaluate_core(spec, frame, None)
    base_dates = pd.to_datetime(frame["date"]).to_numpy()
    # 真实面板为大写列（load_panel映射），兼容两种写法；为空即拒测，避免静默通过
    price_cols = [c for c in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT",
                              "TURNOVER", "MARKET_CAP",
                              "open", "high", "low", "close", "volume", "amount",
                              "turnover_rate", "total_mv")
                  if c in frame.columns]
    if not price_cols:
        return {"passed": False, "failed_dates": [{"date": "no-price-cols"}],
                "note": "no price columns found, audit refused"}
    from field_catalog import EXTENDED_ONLY_FIELDS, LOCAL_ONLY_FIELDS
    failed = []
    for di in sorted(int(x) for x in cands):
        t = dates[di]
        mut = frame.copy()
        after = pd.to_datetime(mut["date"]) > t
        for c in price_cols:
            noise = rng.normal(loc=0.0, scale=1.0, size=int(after.sum()))
            mut.loc[after, c] = noise
        # 衍生列缓存盲区修复：base 已把 9 个本地/扩展列算进 frame 并被 copy 继承；
        # ensure_* 幂等判断会跳过重算，使依赖 VWAP/IND_RET/BETA20 等的前视作弊逃逸。
        # 扰动后强制删除，令 _evaluate_core 从扰动价格重算。
        mut.drop(columns=list(LOCAL_ONLY_FIELDS) + list(EXTENDED_ONLY_FIELDS),
                 errors="ignore", inplace=True)
        try:
            got = _evaluate_core(spec, mut, None)
        except Exception as e:
            failed.append({"date": str(pd.Timestamp(t).date()), "error": str(e)[:100]})
            continue
        keep = base_dates <= t
        a, b = np.asarray(base, dtype=float)[keep], np.asarray(got, dtype=float)[keep]
        if not np.allclose(a, b, equal_nan=True):
            diff = int((~(np.isclose(a, b, equal_nan=True))).sum())
            failed.append({"date": str(pd.Timestamp(t).date()), "n_diff": diff})
    return {"passed": not failed, "failed_dates": failed,
            "note": f"{n_probes} probes, seed={seed}"}


def load_all_custom_specs() -> dict[str, CustomFactorSpec]:
    """自动导入 custom_factors/ 目录下所有模块以触发装饰器注册。"""
    import importlib
    import pkgutil
    from pathlib import Path
    pkg_dir = Path(__file__).resolve().parent / 'custom_factors'
    if pkg_dir.exists():
        for _, mod_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
            importlib.import_module(f'custom_factors.{mod_name}')
    return dict(REGISTRY)
