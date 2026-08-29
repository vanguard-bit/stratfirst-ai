"""Contract tests: REQUIRED_KEYS ⊆ warm build_state; yaml ↔ code alignment."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from features.bar_state import build_state
from strategies.contracts import (
    REQUIRED_KEYS,
    assert_required_keys_complete,
    yaml_code_alignment,
)
from strategies.registry import all_strategy_ids, build_strategy

IST = ZoneInfo("Asia/Kolkata")


def _warm_frame(n: int = 260, *, n_symbols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 09:15", periods=n, freq="1D", tz=IST)
    rows = []
    for j, sym in enumerate(["RELIANCE", "TCS", "INFY"][:n_symbols]):
        base = 1000.0 + j * 100
        for i, ts in enumerate(idx):
            px = base + i * 0.5
            rows.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    "close": px,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_required_keys_cover_all_strategy_ids():
    assert_required_keys_complete()
    assert len(REQUIRED_KEYS) == 21


def test_yaml_matches_built_strategy_metadata():
    for row in yaml_code_alignment():
        assert row["tf_ok"], row
        assert row["product_ok"], row
        assert row["cluster_ok"], row


@pytest.mark.parametrize("sid", all_strategy_ids())
def test_warm_build_state_covers_required_keys(sid):
    df = _warm_frame()
    strat = build_strategy(sid)
    state = build_state(
        df,
        "RELIANCE",
        timeframe=strat.timeframe,
        universe_bars=df,
        now=df["ts"].iloc[-1],
    )
    missing = REQUIRED_KEYS[sid] - set(state.keys())
    assert not missing, f"{sid} missing keys {missing}"
