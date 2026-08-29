"""Backfill history tests (mocked network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_backfill_history_writes_1d(monkeypatch, tmp_path: Path):
    from data.ingest import backfill_history as bh

    def fake_eod(symbol, start, end):
        return pd.DataFrame(
            {
                "date": pd.date_range(start, periods=5, freq="B"),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
            }
        )

    monkeypatch.setattr(bh, "fetch_historical_eod", fake_eod)
    stats = bh.backfill_history(
        ["RELIANCE"],
        years=1,
        db_path=tmp_path / "m.duckdb",
        tfs=("1d", "1W"),
        progress=lambda _m: None,
    )
    assert stats.get("1d", 0) >= 5
    assert stats.get("1W", 0) >= 1


def test_backfill_history_writes_1m(monkeypatch, tmp_path: Path):
    from data.ingest import backfill_history as bh
    from data.ingest.store import DataStore

    def fake_1m(symbol, start, end, *, sleep_s=0.0, progress=None):
        ts = pd.date_range("2024-01-02 09:15", periods=20, freq="min")
        return pd.DataFrame(
            {
                "ts": ts,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            }
        )

    monkeypatch.setattr(bh, "fetch_fyers_1m", fake_1m)
    stats = bh.backfill_history(
        ["RELIANCE"],
        years=1,
        db_path=tmp_path / "m1.duckdb",
        tfs=("1m",),
        skip_eod=True,
        progress=lambda _m: None,
    )
    assert stats.get("1m", 0) >= 1
    with DataStore(db_path=tmp_path / "m1.duckdb") as store:
        store.init_schema()
        assert len(store.read_bars_1m("RELIANCE")) >= 1


def test_resample_5m_from_1m():
    from experiments.strategy_replay import _resample_ohlcv

    ts = pd.date_range("2024-01-02 09:15", periods=15, freq="min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["RELIANCE"] * 15,
            "open": range(15),
            "high": range(15),
            "low": range(15),
            "close": range(15),
            "volume": [1.0] * 15,
        }
    )
    out = _resample_ohlcv(df, "5min")
    assert len(out) == 3
    assert float(out.iloc[0]["volume"]) == 5.0
