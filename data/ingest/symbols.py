from __future__ import annotations

from nse_trader.config import load_yaml


def load_nifty50_symbols() -> list[str]:
    data = load_yaml("nifty50.yaml")
    return list(data.get("symbols", []))


def to_fyers_symbol(symbol: str) -> str:
    """RELIANCE → NSE:RELIANCE-EQ"""
    sym = symbol.upper().strip()
    if sym.startswith("NSE:"):
        return sym
    return f"NSE:{sym}-EQ"


def from_fyers_symbol(fyers_symbol: str) -> str:
    """NSE:RELIANCE-EQ → RELIANCE"""
    sym = fyers_symbol.strip()
    if sym.startswith("NSE:"):
        sym = sym[4:]
    if sym.endswith("-EQ"):
        sym = sym[:-3]
    return sym
