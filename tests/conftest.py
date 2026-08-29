from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from nse_trader.config import ROOT
from sim.fees.refresh import refresh_registry
from sim.friction.measured import Quote
from tests.plan.status import phase_ready

pytest_plugins: list[str] = []


@pytest.fixture(scope="session")
def registry_path() -> Path:
    reg = refresh_registry(offline=True)
    path = ROOT / "data" / "fees" / "registry.json"
    reg.save(path)
    return path


@pytest.fixture
def sample_quote() -> Quote:
    return Quote("RELIANCE", ltp=2500.0, bid=2499.5, ask=2500.5, timestamp="2026-08-10T10:00:00+05:30")


@pytest.fixture
def portfolio_config() -> dict:
    with (ROOT / "config" / "portfolio.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def strategies_config() -> dict:
    with (ROOT / "config" / "strategies.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def ops_config() -> dict:
    with (ROOT / "config" / "ops.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def circuit_locked_uc() -> dict:
    """Stock at upper circuit — LTP equals UC."""
    return {
        "symbol": "SMALLCAP",
        "ltp": 100.0,
        "uc": 100.0,
        "lc": 80.0,
        "bid": 100.0,
        "ask": None,
    }


@pytest.fixture
def mis_square_off_time() -> datetime:
    return datetime(2026, 8, 10, 15, 20, 0)


def pytest_collection_modifyitems(config, items):
    """Auto-skip phaseN tests when that phase is not implemented."""
    for item in items:
        for mark in item.iter_markers():
            if mark.name.startswith("phase") and not phase_ready(mark.name):
                item.add_marker(
                    pytest.mark.skip(reason=f"{mark.name} not implemented yet")
                )
