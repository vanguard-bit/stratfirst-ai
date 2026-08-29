"""Phase 2 contract — exchange rules (circuits, sessions)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase2


def test_buy_rejected_at_upper_circuit(circuit_locked_uc):
    from sim.exchange.rules import evaluate_order

    result = evaluate_order(
        side="BUY",
        order_type="MARKET",
        ltp=circuit_locked_uc["ltp"],
        uc=circuit_locked_uc["uc"],
        lc=circuit_locked_uc["lc"],
        bid=circuit_locked_uc["bid"],
        ask=circuit_locked_uc["ask"],
    )
    assert result.allowed is False
    assert "upper circuit" in result.reason.lower() or "circuit" in result.reason.lower()


def test_sell_allowed_at_upper_circuit(circuit_locked_uc):
    from sim.exchange.rules import evaluate_order

    result = evaluate_order(
        side="SELL",
        order_type="MARKET",
        ltp=circuit_locked_uc["ltp"],
        uc=circuit_locked_uc["uc"],
        lc=circuit_locked_uc["lc"],
        bid=circuit_locked_uc["bid"],
        ask=circuit_locked_uc["ask"],
    )
    assert result.allowed is True


def test_limit_buy_above_uc_rejected():
    from sim.exchange.rules import evaluate_order

    result = evaluate_order(
        side="BUY",
        order_type="LIMIT",
        limit_price=105.0,
        ltp=100.0,
        uc=100.0,
        lc=80.0,
        bid=99.5,
        ask=100.0,
    )
    assert result.allowed is False


def test_market_rejected_when_no_ask():
    from sim.exchange.rules import evaluate_order

    result = evaluate_order(
        side="BUY",
        order_type="MARKET",
        ltp=100.0,
        uc=120.0,
        lc=80.0,
        bid=100.0,
        ask=None,
    )
    assert result.allowed is False


def test_outside_session_rejected(mis_square_off_time):
    from sim.exchange.sessions import is_trading_session

    # 03:00 IST — not a trading session
    assert is_trading_session("2026-08-10T03:00:00+05:30") is False
    assert is_trading_session("2026-08-10T10:00:00+05:30") is True
