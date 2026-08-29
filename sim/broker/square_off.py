from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class SquareOffAction:
    symbol: str
    side: str
    quantity: int
    product: str
    reason: str


class SquareOffEngine:
    """Zerodha-style MIS auto square-off at configured time (default 15:20 IST)."""

    def __init__(self, square_off_time: str = "15:20"):
        hour, minute = (int(part) for part in square_off_time.split(":"))
        self._square_off = time(hour, minute)

    def actions_at(
        self,
        ts: datetime,
        positions: dict[str, dict],
    ) -> list[SquareOffAction]:
        dt = ts if ts.tzinfo else ts.replace(tzinfo=IST)
        dt = dt.astimezone(IST)

        if dt.weekday() >= 5 or dt.time() < self._square_off:
            return []

        actions: list[SquareOffAction] = []
        for symbol, pos in positions.items():
            product = str(pos.get("product", "")).upper()
            if product != "MIS":
                continue
            qty = int(pos.get("qty", 0))
            if qty == 0:
                continue
            if qty > 0:
                actions.append(
                    SquareOffAction(
                        symbol=symbol,
                        side="SELL",
                        quantity=qty,
                        product="MIS",
                        reason="mis_eod_square_off",
                    )
                )
            else:
                actions.append(
                    SquareOffAction(
                        symbol=symbol,
                        side="BUY",
                        quantity=abs(qty),
                        product="MIS",
                        reason="mis_eod_square_off",
                    )
                )
        return actions
