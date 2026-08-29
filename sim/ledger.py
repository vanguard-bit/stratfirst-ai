from __future__ import annotations

from dataclasses import dataclass, field

from sim.events import FillEvent


@dataclass
class VirtualBook:
    strategy_id: str
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0

    @property
    def equity(self) -> float:
        return self.cash + self.realized_pnl

    def apply(self, event: FillEvent) -> None:
        if event.strategy_id != self.strategy_id:
            raise ValueError(
                f"FillEvent strategy {event.strategy_id!r} != book {self.strategy_id!r}"
            )

        side = event.side.upper()
        if side == "BUY":
            self._apply_buy(event)
        elif side == "SELL":
            self._apply_sell(event)
        else:
            raise ValueError(f"unknown side: {event.side!r}")

    def _apply_buy(self, event: FillEvent) -> None:
        self.cash -= event.notional + event.fees
        prev_qty = self.positions.get(event.symbol, 0)
        prev_avg = self.avg_cost.get(event.symbol, 0.0)
        new_qty = prev_qty + event.quantity
        self.positions[event.symbol] = new_qty
        self.avg_cost[event.symbol] = (
            (prev_avg * prev_qty + event.fill_price * event.quantity) / new_qty
        )

    def _apply_sell(self, event: FillEvent) -> None:
        prev_qty = self.positions.get(event.symbol, 0)
        if event.quantity > prev_qty:
            raise ValueError(
                f"cannot sell {event.quantity} {event.symbol} — only {prev_qty} held"
            )
        cost_basis = self.avg_cost.get(event.symbol, 0.0)
        self.realized_pnl += (event.fill_price - cost_basis) * event.quantity
        self.cash += event.notional - event.fees
        new_qty = prev_qty - event.quantity
        if new_qty == 0:
            del self.positions[event.symbol]
            del self.avg_cost[event.symbol]
        else:
            self.positions[event.symbol] = new_qty


@dataclass
class CombinedPortfolio:
    total_capital: float
    weights: dict[str, float]
    virtual_books: dict[str, VirtualBook] = field(default_factory=dict)

    def combined_return(self, returns: dict[str, float]) -> float:
        return sum(self.weights.get(s, 0.0) * returns.get(s, 0.0) for s in returns)

    def effective_capital(self, strategy_id: str) -> float:
        return self.total_capital * self.weights.get(strategy_id, 0.0)
