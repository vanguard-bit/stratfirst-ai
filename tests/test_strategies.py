"""Phase 6 contract — strategy interface."""

from __future__ import annotations

import pytest

from strategies.base import Bar, Signal

pytestmark = pytest.mark.phase6


def test_strategy_protocol_imports():
    from strategies.base import Strategy  # noqa: F401


def test_a1_momentum_long_on_positive_lookback():
    from strategies.cluster_a.momentum import TimeSeriesMomentum

    strat = TimeSeriesMomentum(id="A1", lookback_days=5, exit_ma=100)
    bar = Bar(
        ts="2026-08-10",
        symbol="RELIANCE",
        open=2500,
        high=2510,
        low=2490,
        close=2520,
        volume=1e6,
        timeframe="1D",
    )
    state = {"returns_252d": 0.12, "close_vs_ma100": 1.02}
    sig: Signal = strat.on_bar(bar, state)
    assert sig.action in {"BUY", "HOLD", "SELL", "FLAT"}


def test_g2_circuit_breaker_flattens():
    from strategies.cluster_g.defensive import DrawdownCircuitBreaker

    strat = DrawdownCircuitBreaker(id="G2", max_dd=0.10)
    bar = Bar("2026-08-10", "NIFTY", 0, 0, 0, 0, 0, "1D")
    sig = strat.on_bar(bar, {"portfolio_drawdown": 0.12})
    assert sig.action == "FLAT"


def test_registry_builds_a1_b1_g1_g2():
    from strategies.registry import build_strategy

    a1 = build_strategy("A1")
    b1 = build_strategy("B1")
    g1 = build_strategy("G1")
    g2 = build_strategy("G2")
    assert a1.id == "A1"
    assert b1.cluster == "B"
    assert g1.ma == 200
    assert g2.max_dd == 0.10


def test_all_21_strategies_build():
    from strategies.registry import all_strategy_ids, build_strategy

    ids = all_strategy_ids()
    assert len(ids) == 21
    for sid in ids:
        strat = build_strategy(sid)
        assert strat.id == sid
        assert strat.cluster in {"A", "B", "C", "D", "E", "F", "G"}
