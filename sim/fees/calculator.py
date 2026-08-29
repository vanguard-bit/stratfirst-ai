from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sim.fees.sources.official import FeeRegistry, FeeComponent


@dataclass
class TradeCharges:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    ipft: float = 0.0
    gst: float = 0.0
    total: float = 0.0
    breakdown: dict[str, float] | None = None


class FeeCalculator:
    """Apply fees from registry — no hardcoded regulatory rates."""

    def __init__(self, registry: FeeRegistry):
        self.registry = registry

    @classmethod
    def from_file(cls, path: Path) -> FeeCalculator:
        return cls(FeeRegistry.load(path))

    def _component(self, segment: str, name: str, side: str) -> FeeComponent | None:
        seg = self.registry.segments.get(segment)
        if not seg:
            return None
        for c in seg.components:
            if c.name == name and c.side in {side, "both"}:
                return c
        for c in seg.components:
            if c.name == name:
                return c
        return None

    def _apply_rate(self, comp: FeeComponent | None, turnover: float) -> float:
        if comp is None:
            return 0.0
        if comp.rate_unit == "percent":
            return turnover * comp.rate / 100.0
        if comp.rate_unit == "per_crore":
            return turnover * comp.rate / 1e7
        if comp.rate_unit == "flat_inr":
            return comp.rate
        if comp.rate_unit == "flat_inr_or_pct":
            return min(comp.rate, turnover * 0.0003)
        return 0.0

    def compute(
        self,
        segment: str,
        side: str,
        turnover: float,
    ) -> TradeCharges:
        brokerage_comp = self._component(segment, "brokerage", side)
        brokerage = 0.0
        if brokerage_comp:
            if brokerage_comp.rate_unit == "flat_inr_or_pct":
                brokerage = min(brokerage_comp.rate, turnover * 0.0003)
            else:
                brokerage = self._apply_rate(brokerage_comp, turnover)

        stt = self._apply_rate(self._component(segment, "stt", side), turnover)
        exchange = self._apply_rate(self._component(segment, "exchange_txn", side), turnover)
        sebi = self._apply_rate(self._component(segment, "sebi", side), turnover)
        stamp = self._apply_rate(self._component(segment, "stamp_duty", side), turnover) if side == "buy" else 0.0
        ipft = self._apply_rate(self._component(segment, "ipft", side), turnover)

        taxable = brokerage + exchange + sebi + ipft
        gst_comp = self._component(segment, "gst", side)
        gst = taxable * (gst_comp.rate / 100.0) if gst_comp else 0.0

        breakdown = {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_txn": exchange,
            "sebi": sebi,
            "stamp_duty": stamp,
            "ipft": ipft,
            "gst": gst,
        }
        total = sum(breakdown.values())
        return TradeCharges(total=total, breakdown=breakdown, **breakdown)

    def registry_age_hours(self) -> float:
        if not self.registry.updated_at:
            return float("inf")
        updated = datetime.fromisoformat(self.registry.updated_at.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - updated
        return delta.total_seconds() / 3600.0
