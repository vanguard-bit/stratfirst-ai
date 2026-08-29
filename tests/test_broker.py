"""Phase 3 contract — broker RMS and MIS square-off."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase3


def test_mis_long_squared_off_at_1520(mis_square_off_time):
    from sim.broker.square_off import SquareOffEngine

    engine = SquareOffEngine(square_off_time="15:20")
    positions = {"RELIANCE": {"qty": 10, "product": "MIS"}}
    actions = engine.actions_at(mis_square_off_time, positions)
    assert len(actions) == 1
    assert actions[0].side == "SELL"
    assert actions[0].symbol == "RELIANCE"
    assert actions[0].reason == "mis_eod_square_off"


def test_cnc_not_squared_off_at_1520(mis_square_off_time):
    from sim.broker.square_off import SquareOffEngine

    engine = SquareOffEngine(square_off_time="15:20")
    positions = {"RELIANCE": {"qty": 10, "product": "CNC"}}
    actions = engine.actions_at(mis_square_off_time, positions)
    assert actions == []


def test_insufficient_cash_rejects_buy():
    from sim.broker.rms import validate_order

    result = validate_order(cash=1000, required_cash=50000, product="MIS")
    assert result.allowed is False


def test_square_off_applies_admin_charge(registry_path):
    from sim.broker.charges import admin_square_off_charge

    fee = admin_square_off_charge(registry_path)
    assert fee >= 50  # Zerodha admin square-off baseline


def test_mis_short_cover_buy(mis_square_off_time):
    from sim.broker.square_off import SquareOffEngine

    engine = SquareOffEngine(square_off_time="15:20")
    positions = {"TCS": {"qty": -5, "product": "MIS"}}
    actions = engine.actions_at(mis_square_off_time, positions)
    assert actions[0].side == "BUY"
