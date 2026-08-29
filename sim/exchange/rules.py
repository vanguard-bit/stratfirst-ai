from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExchangeResult:
    allowed: bool
    reason: str = ""


def _at_upper_circuit(ltp: float, uc: float) -> bool:
    return ltp >= uc


def _at_lower_circuit(ltp: float, lc: float) -> bool:
    return ltp <= lc


def evaluate_order(
    *,
    side: str,
    order_type: str,
    ltp: float,
    uc: float,
    lc: float,
    bid: float | None,
    ask: float | None,
    limit_price: float | None = None,
) -> ExchangeResult:
    """NSE-style pre-trade checks: circuits, limit bounds, quote availability."""
    side_u = side.upper()
    type_u = order_type.upper()

    if type_u == "LIMIT":
        if limit_price is None:
            return ExchangeResult(False, "limit order requires limit_price")
        if side_u == "BUY" and limit_price > uc:
            return ExchangeResult(False, "limit buy price above upper circuit")
        if side_u == "SELL" and limit_price < lc:
            return ExchangeResult(False, "limit sell price below lower circuit")

    if side_u == "BUY":
        if _at_upper_circuit(ltp, uc):
            return ExchangeResult(False, "buy rejected — stock at upper circuit")
        if type_u == "MARKET" and ask is None:
            return ExchangeResult(False, "market buy rejected — no ask quote")
    elif side_u == "SELL":
        if _at_lower_circuit(ltp, lc):
            return ExchangeResult(False, "sell rejected — stock at lower circuit")
        if type_u == "MARKET" and bid is None:
            return ExchangeResult(False, "market sell rejected — no bid quote")
    else:
        return ExchangeResult(False, f"unknown side {side!r}")

    return ExchangeResult(True, "accepted")
