"""Circuit band resolution for paper sim."""

from __future__ import annotations

from sim.friction.measured import Quote


def resolve_circuits(
    quote: Quote,
    *,
    prev_close: float | None = None,
    fallback_pct: float = 0.10,
) -> tuple[float | None, float | None]:
    """
    Return (uc, lc).

    1) Fyers upper_ckt/lower_ckt when both > 0 and uc >= lc
    2) Else prev_close ± fallback_pct (quote.prev_close or arg)
    3) Else (None, None) — caller must not invent ltp*1.2
    """
    uc = quote.upper_ckt
    lc = quote.lower_ckt
    if (
        uc is not None
        and lc is not None
        and float(uc) > 0
        and float(lc) > 0
        and float(uc) >= float(lc)
    ):
        return float(uc), float(lc)

    pc = prev_close if prev_close is not None else quote.prev_close
    if pc is not None and float(pc) > 0 and fallback_pct >= 0:
        p = float(pc)
        band = float(fallback_pct)
        return p * (1.0 + band), p * (1.0 - band)

    return None, None


def circuits_or_open(
    quote: Quote,
    *,
    prev_close: float | None = None,
    fallback_pct: float = 0.10,
) -> tuple[float, float]:
    """UC/LC for pipeline.process — open (+inf/-inf) when unresolved."""
    uc, lc = resolve_circuits(quote, prev_close=prev_close, fallback_pct=fallback_pct)
    return (
        float(uc) if uc is not None else float("inf"),
        float(lc) if lc is not None else float("-inf"),
    )


def at_circuit(
    ltp: float,
    uc: float | None,
    lc: float | None,
) -> bool:
    if uc is not None and ltp >= float(uc):
        return True
    if lc is not None and ltp <= float(lc):
        return True
    return False
