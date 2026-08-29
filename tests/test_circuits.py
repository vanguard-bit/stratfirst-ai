"""Circuit resolution tests."""

from __future__ import annotations

import pytest

from sim.exchange.circuits import at_circuit, circuits_or_open, resolve_circuits
from sim.friction.measured import Quote

pytestmark = pytest.mark.phase2


def test_fyers_ckt_wins_over_prev_close():
    q = Quote("X", ltp=100.0, bid=99.0, ask=101.0, upper_ckt=110.0, lower_ckt=90.0, prev_close=100.0)
    uc, lc = resolve_circuits(q, fallback_pct=0.10)
    assert uc == 110.0 and lc == 90.0


def test_zero_fyers_falls_back_to_prev_close_band():
    q = Quote("X", ltp=100.0, bid=99.0, ask=101.0, upper_ckt=0.0, lower_ckt=0.0, prev_close=200.0)
    uc, lc = resolve_circuits(q, fallback_pct=0.10)
    assert uc == pytest.approx(220.0)
    assert lc == pytest.approx(180.0)


def test_missing_prev_returns_none():
    q = Quote("X", ltp=100.0, bid=99.0, ask=101.0)
    assert resolve_circuits(q) == (None, None)


def test_circuits_or_open_uses_inf_when_unresolved():
    q = Quote("X", ltp=100.0, bid=99.0, ask=101.0)
    uc, lc = circuits_or_open(q)
    assert uc == float("inf") and lc == float("-inf")


def test_at_circuit_detects_uc_and_lc():
    assert at_circuit(110.0, 110.0, 90.0) is True
    assert at_circuit(90.0, 110.0, 90.0) is True
    assert at_circuit(100.0, 110.0, 90.0) is False
    assert at_circuit(100.0, None, None) is False
