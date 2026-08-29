"""Fixed-notional universe aggregation + cache helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.strategy_replay import (
    aggregate_fixed_notional_book,
    default_notional_per_symbol,
)


def test_aggregate_fixed_notional_book_math():
    # 2 symbols, one strategy, same day
    # ret_A=0.10 on N0=100 → pnl 10; ret_B=0.00 → pnl 0
    # book ret = 10 / (100*2) = 0.05
    per = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "strategy_id": ["A1", "A1"],
            "symbol": ["AAA", "BBB"],
            "ret": [0.10, 0.00],
            "pnl": [10.0, 0.0],
        }
    )
    book = aggregate_fixed_notional_book(per, n_universe=2, notional_per_symbol=100.0)
    assert len(book) == 1
    assert abs(float(book.iloc[0]["ret"]) - 0.05) < 1e-12


def test_missing_symbol_still_in_denominator():
    # Only one symbol present; n_universe=2 → drag
    per = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "strategy_id": ["A1"],
            "symbol": ["AAA"],
            "ret": [0.10],
            "pnl": [10.0],
        }
    )
    book = aggregate_fixed_notional_book(per, n_universe=2, notional_per_symbol=100.0)
    assert abs(float(book.iloc[0]["ret"]) - 0.05) < 1e-12


def test_default_notional_from_portfolio():
    n0 = default_notional_per_symbol(50)
    assert abs(n0 - 40_000.0) < 1e-6


def test_export_and_replay_two_symbols(tmp_path: Path, monkeypatch):
    """Tiny end-to-end: fake cache + 1 strategy stub via monkeypatch bars."""
    from experiments import strategy_replay as sr

    cache = tmp_path / "cache"
    for sym in ("AAA", "BBB"):
        d = cache / sym
        d.mkdir(parents=True)
        bars = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=5, freq="B"),
                "symbol": sym,
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000] * 5,
            }
        )
        bars["ts"] = pd.to_datetime(bars["date"])
        bars.to_parquet(d / "1d.parquet", index=False)
        bars.to_parquet(d / "1w.parquet", index=False)
        # empty 1m
        pd.DataFrame(
            columns=["ts", "symbol", "open", "high", "low", "close", "volume"]
        ).to_parquet(d / "1m.parquet", index=False)

    # Only enable A1 (1D) by restricting strategy_ids and stubbing registry load
    from dataclasses import dataclass
    from strategies.base import Bar, Signal

    @dataclass
    class Stub:
        id: str = "A1"
        cluster: str = "A"
        timeframe: str = "1D"
        product: str = "CNC"

        def on_bar(self, bar: Bar, state: dict) -> Signal:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0, confidence=1.0)

    monkeypatch.setattr(sr, "load_enabled_strategies", lambda: {"A1": Stub()})
    monkeypatch.setattr(
        sr,
        "load_yaml",
        lambda name: {"strategies": {"A1": {"timeframe": "1D", "enabled": True}}}
        if name == "strategies.yaml"
        else sr.load_yaml(name),
    )

    book = sr.replay_universe_book_returns(
        ["AAA", "BBB"],
        cache_dir=cache,
        workers=1,
        notional_per_symbol=100.0,
        strategy_ids=["A1"],
        export=False,
        progress=lambda _m: None,
    )
    assert not book.empty
    assert set(book.columns) >= {"date", "strategy_id", "ret"}
    assert (book["strategy_id"] == "A1").all()
