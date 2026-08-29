"""Aggregate test health report for coding agents."""

from __future__ import annotations

import pytest

from tests.plan.status import PHASE_IMPLEMENTED

pytestmark = pytest.mark.phase0


class TestPlanStatus:
    def test_phase0_marked_implemented(self):
        assert PHASE_IMPLEMENTED["phase0"] is True

    @pytest.mark.parametrize("phase", ["phase1", "phase2", "phase3", "phase4"])
    def test_future_phases_tracked(self, phase: str):
        assert phase in PHASE_IMPLEMENTED
