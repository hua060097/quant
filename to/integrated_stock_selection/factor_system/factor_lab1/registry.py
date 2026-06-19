"""因子注册中心。"""
from __future__ import annotations

from typing import Dict, Iterable, List

from .base import FactorDefinition


class FactorRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, FactorDefinition] = {}

    def register(self, definition: FactorDefinition) -> None:
        name = definition.name
        if name in self._registry:
            raise ValueError(f"因子{name}已注册，禁止重复注册。")
        self._registry[name] = definition

    def update(self, definition: FactorDefinition) -> None:
        self._registry[definition.name] = definition

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def get(self, name: str) -> FactorDefinition:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"未找到因子{name}，请确认是否已注册。") from exc

    def remove(self, name: str) -> None:
        self._registry.pop(name, None)

    def names(self) -> List[str]:
        return sorted(self._registry.keys())

    def definitions(self) -> Iterable[FactorDefinition]:
        return self._registry.values()


registry = FactorRegistry()


def register_factor(definition: FactorDefinition) -> FactorDefinition:
    registry.register(definition)
    return definition


__all__ = ["registry", "register_factor", "FactorRegistry"]
