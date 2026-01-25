"""因子实验室基础类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Iterable, Any

import pandas as pd


Calculator = Callable[[pd.DataFrame, "FactorContext"], pd.Series]


@dataclass(slots=True)
class FactorContext:
    """携带辅助数据的上下文。"""

    market_data: Optional[pd.DataFrame] = None
    sector_data: Optional[pd.DataFrame] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def require_market(self) -> pd.DataFrame:
        if self.market_data is None:
            raise ValueError("当前因子需要market_data，但未提供。")
        return self.market_data

    def require_sector(self) -> pd.DataFrame:
        if self.sector_data is None:
            raise ValueError("当前因子需要sector_data，但未提供。")
        return self.sector_data


@dataclass(slots=True)
class FactorDefinition:
    """描述单个因子的定义。"""

    name: str
    calculator: Calculator
    required_columns: Set[str] = field(default_factory=set)
    optional_columns: Set[str] = field(default_factory=set)
    lookback_days: int = 0
    requires_market_data: bool = False
    requires_sector_data: bool = False
    description: str = ""

    def run(self, data: pd.DataFrame, context: Optional[FactorContext] = None) -> pd.Series:
        ctx = context or FactorContext()
        missing = self.required_columns.difference(data.columns)
        if missing:
            raise ValueError(f"因子{self.name}缺少必要列: {sorted(missing)}")
        return self.calculator(data, ctx)

    def dependencies(self) -> Iterable[str]:
        # 兼容旧接口，占位
        return []

