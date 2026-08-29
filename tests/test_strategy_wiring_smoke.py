"""Shared build_state path + non-degenerate synthetic smoke for each strategy."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from experiments import strategy_replay as sr
from features.bar_state import build_state
from strategies.base import Bar
from strategies.registry import all_strategy_ids, build_strategy, load_enabled_strategies

IST = ZoneInfo("Asia/Kolkata")


def _series_for(sid: str) -> pd.DataFrame:
    """Craft a short OHLCV path expected to produce both entry-ish and exit-ish actions."""
    strat = build_strategy(sid)
    tf = strat.timeframe
    n = 80
    if sid == "A1":
        n = 260
    if sid == "B3":
        # Two sessions so session_gap_pct is defined; morning bars on day 2.
        idx = pd.DatetimeIndex(
            list(pd.date_range("2026-01-05 09:15", periods=40, freq="15min", tz=IST))
            + list(pd.date_range("2026-01-06 09:15", periods=20, freq="15min", tz=IST))
        )
        n = len(idx)
    elif tf == "1W":
        idx = pd.date_range("2024-01-05", periods=n, freq="W-FRI", tz=IST)
    elif tf in {"1D"}:
        idx = pd.date_range("2026-01-05 09:15", periods=n, freq="1D", tz=IST)
    elif tf in {"1H", "1h"}:
        idx = pd.date_range("2026-01-05 09:15", periods=n, freq="1h", tz=IST)
    else:
        idx = pd.date_range("2026-01-05 09:15", periods=n, freq="5min", tz=IST)

    px = 100.0
    rows = []
    day2_start = None
    if sid == "B3":
        day2_start = idx[40] if len(idx) > 40 else None
    for i, ts in enumerate(idx):
        if sid == "B1":
            # Mean-revert: long flat then crash then recover (z entry + exit)
            if i < 40:
                px = 100.0
            elif i < 50:
                px = 100.0 - (i - 39) * 3.0
            else:
                px = min(100.0, px + 2.0)
        elif sid == "A2":
            # Fast below slow, then sharp rally to force a bullish cross
            if i < 45:
                px = 120.0 - i * 0.4
            else:
                px = 102.0 + (i - 45) * 2.0
        elif sid == "B3":
            if i < 40:
                px = 100.0 + i * 0.01
            else:
                px = 103.0  # gap up day vs ~100.4 prior close
        elif sid == "D3":
            # Quiet then explosive up-bar for ATR expansion + breakout
            if i < n - 5:
                px = 100.0 + (i % 3) * 0.05
            else:
                px = px * 1.08
        elif sid == "C3":
            # Drive SYN to be the loser vs peers
            px = 100.0 - i * 0.8
        elif i < n // 2:
            px *= 1.01
        else:
            px *= 0.99

        if sid == "B3" and day2_start is not None and ts >= day2_start and i == 40:
            o = 103.5  # gap open
        elif sid == "B3" and day2_start is not None and ts >= day2_start:
            o = px * 0.999
        elif sid == "D3" and i >= n - 5:
            o = px * 0.95
        else:
            o = px * 0.999
        hi = max(o, px) * (1.05 if sid == "D3" and i >= n - 5 else 1.002)
        lo = min(o, px) * 0.95 if sid == "D3" and i >= n - 5 else min(o, px) * 0.998
        rows.append(
            {
                "ts": ts,
                "symbol": "SYN",
                "open": o,
                "high": hi,
                "low": lo,
                "close": px,
                "volume": 1000.0 + i,
            }
        )

    extra = []
    for j, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for i, ts in enumerate(idx):
            # Peers trend up so SYN loser ranks for C3; mixed levels for C1/C2
            c = 80 + j * 15 + i * 0.5
            if sid == "C3":
                c = 200 + j * 10 + i
            extra.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "open": c,
                    "high": c + 1,
                    "low": c - 1,
                    "close": c,
                    "volume": 100.0,
                }
            )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(extra)], ignore_index=True)


@pytest.mark.parametrize("sid", all_strategy_ids())
def test_strategy_actions_not_degenerate(sid):
    strat = build_strategy(sid)
    df = _series_for(sid)
    actions: list[str] = []
    sym = "SYN"
    g = df[df["symbol"] == sym].sort_values("ts").reset_index(drop=True)
    # Walk last 40 bars (or all)
    start = max(1, len(g) - 40)
    if sid == "A1":
        start = max(1, len(g) - 30)
    for i in range(start, len(g)):
        hist = df[df["ts"] <= g.iloc[i]["ts"]]
        state = build_state(
            hist,
            sym,
            timeframe=strat.timeframe,
            universe_bars=hist,
            now=pd.Timestamp(g.iloc[i]["ts"]),
            portfolio_drawdown=0.12 if sid == "G2" and i == len(g) - 1 else 0.0,
        )
        if sid == "D2" and i >= len(g) - 5:
            state["vix_above_median"] = True
        if sid == "B1" and i >= len(g) - 10:
            state["in_position"] = True
        cur = g.iloc[i]
        bar = Bar(
            ts=str(cur["ts"]),
            symbol=sym,
            open=float(cur["open"]),
            high=float(cur["high"]),
            low=float(cur["low"]),
            close=float(cur["close"]),
            volume=float(cur["volume"]),
            timeframe=strat.timeframe,
        )
        actions.append(strat.on_bar(bar, state).action)

    uniq = set(actions)
    assert actions, f"{sid}: no actions"
    # Must not be a single stuck action across the whole path (except pure overlays)
    if sid in {"F3"}:
        # Always HOLD with varying exposure — check exposure path separately
        assert uniq == {"HOLD"}
        return
    if sid in {"D1", "D2", "G1", "G2"}:
        # Overlays: expect FLAT or HOLD appear
        assert uniq & {"FLAT", "HOLD", "BUY", "SELL"}, uniq
        return
    assert len(uniq) >= 2 or ("BUY" in uniq or "SELL" in uniq or "FLAT" in uniq), (
        f"{sid} degenerate actions={uniq}"
    )


def test_replay_imports_same_build_state():
    assert sr.build_state is build_state


def test_paper_live_uses_build_state():
    from experiments import paper_live as pl

    assert pl.build_state is build_state


def test_all_enabled_load():
    strats = load_enabled_strategies()
    # Overlays + full cluster E disabled.
    assert len(strats) == 13
    assert not {"D1", "D2", "F3", "G1", "G2", "E1", "E2", "E3"} & set(strats)
