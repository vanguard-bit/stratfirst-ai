from __future__ import annotations

from dataclasses import dataclass

VALID_MATERIALITY = frozenset({"low", "medium", "high"})


@dataclass
class LLMFeatureRow:
    symbol: str
    as_of: str
    sentiment: float
    materiality: str
    events: list


def validate_row(row: LLMFeatureRow) -> None:
    if not row.symbol:
        raise ValueError("symbol required")
    if not row.as_of:
        raise ValueError("as_of required")
    if not -1.0 <= row.sentiment <= 1.0:
        raise ValueError(f"sentiment must be in [-1, 1], got {row.sentiment}")
    if row.materiality not in VALID_MATERIALITY:
        raise ValueError(f"invalid materiality: {row.materiality!r}")
    if not isinstance(row.events, list):
        raise ValueError("events must be a list")
