"""Parity and speed tests for vectorized build_state_frame."""

from __future__ import annotations

import math
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from features.bar_state import build_state
from features.state_series import build_state_frame, state_row_to_dict

IST = ZoneInfo("Asia/Kolkata")

COMPARE_KEYS = [
    "warmup",
    "rsi",
    "rsi14",
    "zscore",
    "close_vs_ma20",
    "close_vs_ma100",
    "close_vs_ma200",
    "ma_cross",
    "ma_fast",
    "ma_slow",
    "donchian_upper",
    "donchian_lower",
    "returns_20d",
    "returns_252d",
    "returns_lookback_ready",
    "gap_pct",
    "session_gap_pct",
    "adx",
    "realized_vol",
    "atr_ratio",
    "breakout_dir",
    "orb_high",
    "orb_low",
    "orb_complete",
    "vwap_dev",
    "in_power_hour",
    "intraday_mom",
    "day_of_month",
    "in_turn_of_month",
    "day_of_week",
    "expiry_week",
    "universe_size",
    "momentum_rank",
    "vol_quintile",
]


def _synth(n: int = 80, freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-03-02 09:15", periods=n, freq=freq, tz=IST)
    px = 100.0
    rows = []
    for i, ts in enumerate(idx):
        px *= 1.001 if i % 7 else 0.998
        rows.append(
            {
                "ts": ts,
                "symbol": "SYN",
                "open": px * 0.999,
                "high": px * 1.002,
                "low": px * 0.997,
                "close": px,
                "volume": 1000.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _close(a, b, tol=1e-8):
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        if a is None or b is None:
            return False
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            return True
        if pd.isna(a) and pd.isna(b):
            return True
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


@pytest.mark.parametrize("i", [5, 20, 40, 60, 79])
def test_state_frame_parity_synth(i):
    df = _synth(80)
    frame = build_state_frame(df, timeframe="5m")
    hist = df.iloc[: i + 1]
    ref = build_state(hist, "SYN", timeframe="5m", now=df.iloc[i]["ts"])
    got = state_row_to_dict(frame.iloc[i])
    for k in COMPARE_KEYS:
        assert k in got and k in ref, k
        assert _close(got[k], ref[k]), f"i={i} key={k} got={got[k]!r} ref={ref[k]!r}"


def test_state_frame_faster_than_naive_build_state():
    df = _synth(2000)
    # vectorized
    t0 = time.perf_counter()
    frame = build_state_frame(df, timeframe="5m")
    # walk like replay (dict materialization)
    for i in range(1, len(frame)):
        _ = state_row_to_dict(frame.iloc[i])
    dt_fast = time.perf_counter() - t0

    # naive sample: every 50th bar full build_state on hist (extrapolate)
    sample_idx = list(range(50, len(df), 50))[:10]
    t0 = time.perf_counter()
    for i in sample_idx:
        build_state(df.iloc[: i + 1], "SYN", timeframe="5m", now=df.iloc[i]["ts"])
    dt_sample = time.perf_counter() - t0
    # extrapolate full walk cost
    per_bar = dt_sample / max(len(sample_idx), 1)
    dt_naive_est = per_bar * (len(df) - 1)
    speedup = dt_naive_est / max(dt_fast, 1e-9)
    assert speedup >= 20.0, f"speedup={speedup:.1f}x fast={dt_fast:.3f}s naive_est={dt_naive_est:.3f}s"


def test_reliance_cache_parity_if_present():
    from pathlib import Path

    p = Path("data/store/cache/replay_bars/RELIANCE/1m.parquet")
    if not p.exists():
        pytest.skip("no RELIANCE 1m cache")
    from experiments.strategy_replay import _resample_ohlcv

    m1 = pd.read_parquet(p)
    m1["ts"] = pd.to_datetime(m1["ts"])
    bars = _resample_ohlcv(m1.head(3000), "5min")
    if len(bars) < 100:
        pytest.skip("too few 5m bars")
    bars = bars.head(120).reset_index(drop=True)
    frame = build_state_frame(bars, timeframe="5m")
    for i in (30, 60, 90, 110):
        ref = build_state(bars.iloc[: i + 1], "RELIANCE", timeframe="5m", now=bars.iloc[i]["ts"])
        got = state_row_to_dict(frame.iloc[i])
        for k in COMPARE_KEYS:
            assert _close(got[k], ref[k], tol=1e-6), f"i={i} {k} got={got[k]!r} ref={ref[k]!r}"
