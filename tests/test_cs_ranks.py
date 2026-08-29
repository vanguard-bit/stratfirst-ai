from __future__ import annotations

import numpy as np
import pandas as pd

from features.bar_state import build_state
from features.cs_ranks import cross_sectional_ranks


def _univ(n_bars: int = 80) -> pd.DataFrame:
    """AAA rises ~20% over last 20 bars; BBB flat; CCC falls ~10%."""
    rng = pd.date_range("2026-01-02 09:15", periods=n_bars, freq="15min", tz="Asia/Kolkata")
    rows = []
    for i, ts in enumerate(rng):
        # Last 21 closes drive mom: AAA up, CCC down
        aaa = 100.0 * (1.0 + 0.01 * max(0, i - (n_bars - 21)))
        bbb = 200.0  # higher price but flat → must NOT win momentum on price level
        ccc = 50.0 * (1.0 - 0.005 * max(0, i - (n_bars - 21)))
        for sym, px in (("AAA", aaa), ("BBB", bbb), ("CCC", ccc)):
            rows.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_rank_by_return_not_price():
    univ = _univ()
    ranks_aaa = cross_sectional_ranks(univ, "AAA")
    ranks_bbb = cross_sectional_ranks(univ, "BBB")
    ranks_ccc = cross_sectional_ranks(univ, "CCC")
    assert ranks_aaa["momentum_rank"] < ranks_bbb["momentum_rank"]
    assert ranks_ccc["momentum_rank"] > ranks_bbb["momentum_rank"]
    # BBB has highest last close but flat return → not rank 1
    assert ranks_bbb["momentum_rank"] != 1


def test_reversion_rank_prefers_losers():
    univ = _univ()
    assert cross_sectional_ranks(univ, "CCC")["reversion_rank"] == 1
    assert cross_sectional_ranks(univ, "AAA")["reversion_rank"] > 1


def test_beta_rank_present():
    univ = _univ()
    r = cross_sectional_ranks(univ, "AAA")
    assert 1 <= r["beta_rank"] <= 3
    assert r["universe_size"] == 3


def test_build_state_uses_cs_ranks():
    univ = _univ()
    sym = univ[univ["symbol"] == "AAA"]
    state = build_state(sym, "AAA", timeframe="15m", now=sym["ts"].iloc[-1], universe_bars=univ)
    assert state["momentum_rank"] == cross_sectional_ranks(univ, "AAA")["momentum_rank"]
    assert state["momentum_rank"] != 999
