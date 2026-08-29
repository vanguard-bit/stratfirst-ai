from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RmsResult:
    allowed: bool
    reason: str = ""


def validate_order(
    *,
    cash: float,
    required_cash: float,
    product: str,
) -> RmsResult:
    """Broker RMS pre-check — cash/margin availability."""
    _ = product  # reserved for product-specific margin rules in later phases
    if required_cash <= 0:
        return RmsResult(False, "invalid required_cash")
    if cash < required_cash:
        return RmsResult(False, "insufficient cash")
    return RmsResult(True, "accepted")
