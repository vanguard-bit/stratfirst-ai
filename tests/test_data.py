"""Phase 5 contract — data store and backfill."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase5


def test_data_store_schema(tmp_path):
    from data.ingest.store import DataStore

    store = DataStore(db_path=tmp_path / "schema_test.duckdb")
    store.init_schema()
    tables = {
        row[0]
        for row in store.con.execute("SHOW TABLES").fetchall()
    }
    assert "bars_1m" in tables
    assert "bars_1d" in tables
    assert "friction_spreads" in tables
    store.close()


def test_resample_1m_to_15m():
    import pandas as pd
    from data.resample.bars import resample_bars

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-08-10 09:15", periods=30, freq="1min"),
            "symbol": ["RELIANCE"] * 30,
            "open": 2500.0,
            "high": 2501.0,
            "low": 2499.0,
            "close": 2500.0,
            "volume": 1000,
        }
    )
    out = resample_bars(df, "15min")
    assert len(out) >= 2


def test_backfill_writes_eod_bars(tmp_path, monkeypatch):
    import pandas as pd

    from data.ingest import backfill
    from data.ingest.store import DataStore

    db = tmp_path / "test.duckdb"
    mock_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-01", "2026-08-04"]),
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
        }
    )

    def _mock_fetch(symbol: str, start, end):
        return mock_df

    monkeypatch.setattr(backfill, "fetch_historical_eod", _mock_fetch)

    rows = backfill.backfill_eod(["NIFTY 50"], years=1, db_path=db)
    assert rows == 2
    assert db.exists()

    store = DataStore(db_path=db)
    assert store.row_count("bars_1d") == 2
    out = store.read_bars_1d("NIFTY 50")
    store.close()
    assert list(out["close"]) == [101.0, 102.0]


def test_record_spread_snapshot(tmp_path):
    from data.ingest.live import record_spread_snapshot
    from data.ingest.store import DataStore
    from sim.friction.measured import Quote

    db = tmp_path / "test.duckdb"
    q = Quote("RELIANCE", ltp=2500.0, bid=2499.5, ask=2500.5)
    n = record_spread_snapshot([q], db_path=db)
    assert n == 1

    store = DataStore(db_path=db)
    assert store.row_count("friction_spreads") == 1
    store.close()


def test_read_latest_spreads_one_row_per_symbol(tmp_path):
    import pandas as pd

    from data.ingest.store import DataStore

    db = tmp_path / "spreads.duckdb"
    store = DataStore(db_path=db)
    store.init_schema()
    ts_a = pd.Timestamp("2026-08-11 15:29:00")
    ts_b = pd.Timestamp("2026-08-11 15:29:58")
    # Shared timestamps across symbols previously pulled stale rows into "latest".
    df = pd.DataFrame(
        [
            {
                "ts": ts_a,
                "symbol": "RELIANCE",
                "bid": 2499.5,
                "ask": 2500.5,
                "ltp": 2500.0,
                "half_spread_bps": 2.0,
            },
            {
                "ts": ts_b,
                "symbol": "RELIANCE",
                "bid": 1362.0,
                "ask": 1282.0,
                "ltp": 1320.0,
                "half_spread_bps": -300.0,
            },
            {
                "ts": ts_b,
                "symbol": "TMPV",
                "bid": 347.0,
                "ask": 347.2,
                "ltp": 347.1,
                "half_spread_bps": 2.8,
            },
            {
                "ts": ts_a,
                "symbol": "TMPV",
                "bid": 340.0,
                "ask": 341.0,
                "ltp": 340.5,
                "half_spread_bps": 14.0,
            },
        ]
    )
    store.write_friction_spreads(df)
    latest = store.read_latest_spreads()
    assert len(latest) == 2
    by = {str(r.symbol): r for r in latest.itertuples()}
    assert float(by["RELIANCE"].bid) == 1362.0  # newest row even if crossed
    valid = store.read_latest_spreads(valid_only=True)
    # Latest *valid* RELIANCE is the older uncrossed row; TMPV newest is valid.
    by_v = {str(r.symbol): r for r in valid.itertuples()}
    assert set(by_v) == {"RELIANCE", "TMPV"}
    assert float(by_v["RELIANCE"].ask) == 2500.5
    assert float(by_v["TMPV"].ask) == 347.2
    store.close()


def test_parse_quote_rejects_b_a_and_crossed():
    from data.ingest.fyers_ws import _parse_quote

    # Ambiguous short keys alone must not become a Quote
    assert _parse_quote({"symbol": "NSE:TMPV-EQ", "ltp": 347.0, "b": 357.0, "a": 336.0}) is None
    good = _parse_quote(
        {"symbol": "NSE:TMPV-EQ", "ltp": 347.0, "bid_price": 346.9, "ask_price": 347.1}
    )
    assert good is not None
    assert good.bid == 346.9 and good.ask == 347.1
    assert (
        _parse_quote(
            {"symbol": "NSE:TMPV-EQ", "ltp": 347.0, "bid_price": 357.0, "ask_price": 336.0}
        )
        is None
    )
