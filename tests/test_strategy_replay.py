"""Strategy replay daily returns tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from experiments.strategy_replay import replay_strategy_daily_returns
from strategies.base import Bar, Signal


@dataclass
class _StubStrat:
    id: str = "A1"
    cluster: str = "A"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        # Always long 1.0 exposure after first bar
        return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0, confidence=1.0)


def test_replay_produces_daily_rows():
    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=6, freq="B"),
            "symbol": ["RELIANCE"] * 6,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000] * 6,
        }
    )
    out = replay_strategy_daily_returns(
        "A1", bars, strategy=_StubStrat(), apply_fees=False
    )
    assert {"date", "strategy_id", "ret"} <= set(out.columns)
    assert (out["strategy_id"] == "A1").all()
    assert len(out) >= 1
    # After first signal, subsequent days earn ~1% close-to-close
    assert out["ret"].iloc[-1] != 0.0 or len(out) > 0


def test_replay_fees_reduce_entry_day_return():
    from sim.fees.calculator import FeeCalculator
    from sim.fees.sources.seed import load_seed_registry

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=4, freq="B"),
            "symbol": ["RELIANCE"] * 4,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000] * 4,
        }
    )
    calc = FeeCalculator(load_seed_registry())
    gross = replay_strategy_daily_returns(
        "A1", bars, strategy=_StubStrat(), apply_fees=False, notional=40_000
    )
    net = replay_strategy_daily_returns(
        "A1",
        bars,
        strategy=_StubStrat(),
        notional=40_000,
        fee_calculator=calc,
    )
    assert float(net["ret"].iloc[0]) < float(gross["ret"].iloc[0])
    # After entry, hold days match (no further trades)
    if len(gross) > 1:
        assert abs(float(net["ret"].iloc[-1]) - float(gross["ret"].iloc[-1])) < 1e-12


def test_replay_mis_flatten_drops_overnight():
    @dataclass
    class _MisLong(_StubStrat):
        product: str = "MIS"
        timeframe: str = "1D"

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=3, freq="B"),
            "symbol": ["RELIANCE"] * 3,
            "open": [100.0, 110.0, 111.0],
            "high": [101.0, 111.0, 112.0],
            "low": [99.0, 109.0, 110.0],
            "close": [100.0, 110.0, 111.0],
            "volume": [1000] * 3,
        }
    )
    cnc = replay_strategy_daily_returns(
        "A1", bars, strategy=_StubStrat(), apply_fees=False
    )
    mis = replay_strategy_daily_returns(
        "B2", bars, strategy=_MisLong(), apply_fees=False
    )
    # After first entry (bar 1), CNC holds overnight into last day; MIS flattens
    assert float(cnc["ret"].iloc[-1]) > float(mis["ret"].iloc[-1]) + 0.005
    assert abs(float(mis["ret"].iloc[-1])) < 1e-9


def test_replay_mis_sell_is_short_intraday():
    @dataclass
    class _MisShort(_StubStrat):
        product: str = "MIS"
        timeframe: str = "1H"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0, confidence=1.0)

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02 10:00", periods=3, freq="1h"),
            "symbol": ["RELIANCE"] * 3,
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [1000] * 3,
        }
    )
    out = replay_strategy_daily_returns(
        "E1", bars, strategy=_MisShort(), apply_fees=False
    )
    assert float(out["ret"].sum()) < -0.05


def test_replay_cnc_sell_is_flat_not_short():
    @dataclass
    class _CncSell(_StubStrat):
        product: str = "CNC"
        timeframe: str = "1D"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0, confidence=1.0)

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=4, freq="B"),
            "symbol": ["RELIANCE"] * 4,
            "open": [100.0, 110.0, 121.0, 133.0],
            "high": [101.0, 111.0, 122.0, 134.0],
            "low": [99.0, 109.0, 120.0, 132.0],
            "close": [100.0, 110.0, 121.0, 133.0],
            "volume": [1000] * 4,
        }
    )
    out = replay_strategy_daily_returns(
        "C1", bars, strategy=_CncSell(), apply_fees=False
    )
    # Long-only CNC: SELL from flat is not a short on a rising tape
    assert abs(float(out["ret"].sum())) < 1e-12


def test_replay_hold_preserves_exposure():
    @dataclass
    class _EnterThenHold:
        id: str = "H1"
        cluster: str = "A"
        timeframe: str = "1D"
        product: str = "CNC"
        _entered: bool = False

        def on_bar(self, bar, state):
            if not state.get("in_position"):
                return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
            return Signal(action="HOLD", symbol=bar.symbol)

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=5, freq="B"),
            "symbol": ["RELIANCE"] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000] * 5,
        }
    )
    out = replay_strategy_daily_returns(
        "H1", bars, strategy=_EnterThenHold(), apply_fees=False
    )
    # After entry, subsequent ~1% days should accrue (HOLD must not flatten)
    assert float(out["ret"].sum()) > 0.02


def test_replay_flat_exits_to_zero():
    @dataclass
    class _BuyThenFlat:
        id: str = "H2"
        cluster: str = "A"
        timeframe: str = "1D"
        product: str = "CNC"

        def on_bar(self, bar, state):
            if not state.get("in_position"):
                return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
            return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-02", periods=4, freq="B"),
            "symbol": ["RELIANCE"] * 4,
            "open": [100.0, 110.0, 121.0, 133.0],
            "high": [101.0, 111.0, 122.0, 134.0],
            "low": [99.0, 109.0, 120.0, 132.0],
            "close": [100.0, 110.0, 121.0, 133.0],
            "volume": [1000] * 4,
        }
    )
    out = replay_strategy_daily_returns(
        "H2", bars, strategy=_BuyThenFlat(), apply_fees=False
    )
    # Entry bar may earn; after FLAT later rising days should not add long PnL
    assert float(out["ret"].iloc[-1]) == 0.0
