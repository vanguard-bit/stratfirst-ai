"""Regime feature builder tests."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from meta.regime import (
    is_expiry_week,
    last_thursday_of_month,
    build_regime,
    wilder_adx,
    index_ohlc_from_universe,
)

IST = ZoneInfo("Asia/Kolkata")
pytestmark = pytest.mark.runtime


def test_last_thursday_aug_2026():
    assert last_thursday_of_month(2026, 8) == date(2026, 8, 27)


def test_expiry_week_flags():
    assert is_expiry_week(date(2026, 8, 27)) is True  # monthly expiry Thu
    assert is_expiry_week(date(2026, 8, 11)) is True  # Tuesday = weekly expiry
    assert is_expiry_week(date(2026, 8, 12)) is False  # Wed, not near month-end expiry


def test_wilder_adx_trend():
    n = 80
    close = pd.Series([100 + i * 0.5 for i in range(n)])
    high = close + 1
    low = close - 1
    adx = wilder_adx(high, low, close, 14)
    assert adx > 0


def test_build_regime_from_bars(monkeypatch):
    monkeypatch.setattr("meta.regime.fetch_india_vix", lambda **kw: 18.5)
    ts = pd.date_range("2026-08-01 09:15", periods=200, freq="1min", tz=IST)
    rows = []
    for sym in ("RELIANCE", "TCS"):
        for i, t in enumerate(ts):
            px = 1000 + i
            rows.append(
                {
                    "ts": t,
                    "symbol": sym,
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    "close": px,
                    "volume": 100,
                }
            )
    bars = pd.DataFrame(rows)
    regime = build_regime(bars, now=datetime(2026, 8, 11, 14, 0, tzinfo=IST))
    assert regime.vix == 18.5
    assert regime.adx >= 0
    assert isinstance(regime.expiry_week, bool)


def test_index_ohlc_from_universe():
    df = pd.DataFrame(
        {
            "ts": ["2026-08-11 09:15"] * 2,
            "symbol": ["A", "B"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1, 1],
        }
    )
    out = index_ohlc_from_universe(df)
    assert len(out) == 1
    assert out.iloc[0]["close"] == 15.5
