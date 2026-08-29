from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meta.features import RegimeFeatures
from meta.rules_v0 import rules_v0_weights


@dataclass
class AllocatorConstraints:
    max_strategy_weight: float = 0.25
    max_cluster_weight: float = 0.40
    min_cash: float = 0.05
    max_cash: float = 0.30


class MetaAllocator:
    def __init__(
        self,
        strategy_ids: list[str],
        cluster_of: dict[str, str],
        constraints: AllocatorConstraints | None = None,
        mode: str = "equal_weight",
        llm_tilt: dict[str, Any] | None = None,
    ):
        self.strategy_ids = strategy_ids
        self.cluster_of = cluster_of
        self.constraints = constraints or AllocatorConstraints()
        self.mode = mode
        self.llm_tilt = llm_tilt

    def allocate(self, regime: RegimeFeatures) -> dict[str, float]:
        if self.mode == "equal_weight":
            w = 1.0 / len(self.strategy_ids)
            raw = {sid: w for sid in self.strategy_ids}
        elif self.mode == "rules":
            raw = rules_v0_weights(
                self.strategy_ids, self.cluster_of, regime, llm_tilt=self.llm_tilt
            )
        else:
            raw = rules_v0_weights(
                self.strategy_ids, self.cluster_of, regime, llm_tilt=self.llm_tilt
            )
        return self._apply_constraints(raw)

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        c = self.constraints
        clipped = {sid: min(weights.get(sid, 0.0), c.max_strategy_weight) for sid in self.strategy_ids}

        cluster_totals: dict[str, float] = {}
        for sid, w in clipped.items():
            cl = self.cluster_of.get(sid, "?")
            cluster_totals[cl] = cluster_totals.get(cl, 0.0) + w

        for sid, w in list(clipped.items()):
            cl = self.cluster_of.get(sid, "?")
            if cluster_totals.get(cl, 0.0) > c.max_cluster_weight:
                scale = c.max_cluster_weight / cluster_totals[cl]
                clipped[sid] = w * scale

        total = sum(clipped.values())
        if total > 1.0 - c.min_cash:
            scale = (1.0 - c.min_cash) / total
            clipped = {sid: w * scale for sid, w in clipped.items()}

        return clipped
