"""Feature / bar-close helpers for live paper."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from features.bar_state import (
    bar_just_closed,
    build_state,
    closed_timeframes,
    closed_timeframes_for_tick,
    resample_symbol_tf,
)

IST = ZoneInfo("Asia/Kolkata")


def test_bar_just_closed_5m():
    ts = datetime(2026, 8, 11, 9, 19, tzinfo=IST)
    assert bar_just_closed(ts, "5m") is True
    assert bar_just_closed(datetime(2026, 8, 11, 9, 17, tzinfo=IST), "5m") is False


def test_closed_timeframes_includes_1m():
    ts = datetime(2026, 8, 11, 9, 19, tzinfo=IST)
    tfs = closed_timeframes(ts)
    assert "1m" in tfs
    assert "5m" in tfs


def test_bar_just_closed_1d_only_at_1529():
    assert bar_just_closed(datetime(2026, 8, 11, 15, 29, tzinfo=IST), "1D") is True
    assert bar_just_closed(datetime(2026, 8, 11, 15, 30, tzinfo=IST), "1D") is False
    assert bar_just_closed(datetime(2026, 8, 11, 15, 35, tzinfo=IST), "1D") is False


def _bars_until(last_hhmm: str) -> pd.DataFrame:
    last = pd.Timestamp(f"2026-08-13 {last_hhmm}", tz=IST)
    start = pd.Timestamp("2026-08-13 09:15", tz=IST)
    idx = pd.date_range(start, last, freq="1min")
    return pd.DataFrame(
        {
            "ts": idx,
            "symbol": ["RELIANCE"] * len(idx),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )


def test_1d_uses_last_bar_minute_not_wall_clock_after_ingest():
    """Ingest duration ~45s: wall 15:30, last 1m bar still 15:29 → 1D must close."""
    bars = _bars_until("15:29")
    wall = datetime(2026, 8, 13, 15, 30, 12, tzinfo=IST)
    tfs = closed_timeframes_for_tick(wall, bars)
    assert "1D" in tfs
    assert "1D" not in closed_timeframes(wall)


def test_1d_catchup_when_last_bar_is_1530():
    """If the 15:29 tick was skipped, still eval 1D in the 15:29–15:35 window."""
    bars = _bars_until("15:30")
    wall = datetime(2026, 8, 13, 15, 30, 40, tzinfo=IST)
    assert "1D" in closed_timeframes_for_tick(wall, bars)


def test_1d_no_catchup_before_session_last_minute():
    bars = _bars_until("15:28")
    wall = datetime(2026, 8, 13, 15, 30, tzinfo=IST)
    assert "1D" not in closed_timeframes_for_tick(wall, bars)


def test_build_state_has_core_keys():
    idx = pd.date_range("2026-08-11 09:15", periods=40, freq="15min", tz=IST)
    df = pd.DataFrame(
        {
            "ts": idx,
            "symbol": ["RELIANCE"] * 40,
            "open": 2500.0,
            "high": 2510.0,
            "low": 2490.0,
            "close": [2500 + i for i in range(40)],
            "volume": 1000.0,
        }
    )
    state = build_state(df, "RELIANCE", timeframe="15m", now=idx[-1])
    assert "rsi" in state
    assert "zscore" in state
    assert "orb_complete" in state
    assert "donchian_upper" in state
    assert "donchian_lower" in state
    assert "ma_cross" in state
    assert state["warmup"] is False
    # Prior-window Donchian must not collapse to last close
    assert state["donchian_upper"] != float(df["close"].iloc[-1])
    assert state["donchian_upper"] == float(df["high"].iloc[-(20 + 1) : -1].max())


def test_build_state_warmup_short_history():
    idx = pd.date_range("2026-08-11 09:15", periods=10, freq="15min", tz=IST)
    df = pd.DataFrame(
        {
            "ts": idx,
            "symbol": ["RELIANCE"] * 10,
            "open": 2500.0,
            "high": 2510.0,
            "low": 2490.0,
            "close": 2500.0,
            "volume": 1000.0,
        }
    )
    state = build_state(df, "RELIANCE", timeframe="15m", now=idx[-1])
    assert state["warmup"] is True
    assert state["donchian_upper"] != state["donchian_upper"]  # NaN


def test_resample_symbol_tf_15m():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-08-11 09:15", periods=30, freq="1min", tz=IST),
            "symbol": ["TCS"] * 30,
            "open": 3000.0,
            "high": 3001.0,
            "low": 2999.0,
            "close": 3000.0,
            "volume": 100,
        }
    )
    out = resample_symbol_tf(df, "15m")
    assert len(out) >= 2
    assert "close" in out.columns
