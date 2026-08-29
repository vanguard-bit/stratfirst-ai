from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quote:
    symbol: str
    ltp: float
    bid: float | None = None
    ask: float | None = None
    timestamp: str = ""
    upper_ckt: float | None = None
    lower_ckt: float | None = None
    prev_close: float | None = None


class MeasuredFriction:
    """
    Spread/slippage from live or recorded bid-ask — never guessed bps.
    """

    def fill_price(self, quote: Quote, side: str) -> tuple[float, dict[str, float]]:
        if quote.bid is not None and quote.ask is not None and quote.ask >= quote.bid:
            if side.upper() == "BUY":
                return quote.ask, {"fill": quote.ask, "half_spread": (quote.ask - quote.bid) / 2}
            return quote.bid, {"fill": quote.bid, "half_spread": (quote.ask - quote.bid) / 2}

        if quote.bid is not None and quote.ask is not None and quote.ask < quote.bid:
            raise ValueError(
                f"Crossed bid/ask for {quote.symbol} (bid={quote.bid}, ask={quote.ask}). "
                "Refuse fill — friction_mode=measured forbids guessed slippage."
            )
        raise ValueError(
            f"No measured bid/ask for {quote.symbol}. "
            "Record market depth or spread snapshots — friction_mode=measured forbids guessed slippage."
        )

    @staticmethod
    def half_spread_bps(quote: Quote) -> float:
        if quote.bid is None or quote.ask is None or quote.ltp <= 0:
            raise ValueError(f"Cannot measure spread for {quote.symbol}")
        return ((quote.ask - quote.bid) / quote.ltp) * 10_000
