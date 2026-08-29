"""Live multi-strategy paper tick tests."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from sim.friction.measured import Quote

IST = ZoneInfo("Asia/Kolkata")
pytestmark = pytest.mark.runtime


def _bars(n: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2026-08-11 09:15", periods=n, freq="1min", tz=IST)
    rows = []
    for sym, base in (("RELIANCE", 2500.0), ("TCS", 3500.0)):
        for i, t in enumerate(ts):
            px = base + i * 0.5
            rows.append(
                {
                    "ts": t,
                    "symbol": sym,
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    "close": px,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_paper_live_skips_placeholder(tmp_path):
    from experiments.paper_live import run_paper_live_tick

    out = run_paper_live_tick(
        bars_1m=_bars(30),
        quotes=[Quote("RELIANCE", ltp=2500, bid=2499, ask=2501)],
        now=datetime(2026, 8, 11, 9, 19, tzinfo=IST),
        mode="placeholder",
        out_dir=tmp_path,
    )
    assert out["skipped"] is True
    assert out["fills"] == 0


def test_paper_live_tick_runs_strategies(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures

    # Keep books under tmp
    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=30.0, vix=18.0, vix_above_median=True, expiry_week=True
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 / 21 for sid in kw["strategy_ids"]},
    )

    quotes = [
        Quote("RELIANCE", ltp=2560.0, bid=2559.0, ask=2561.0),
        Quote("TCS", ltp=3560.0, bid=3559.0, ask=3561.0),
    ]
    # 09:19 closes a 5m bar
    out = pl.run_paper_live_tick(
        bars_1m=_bars(120),
        quotes=quotes,
        now=datetime(2026, 8, 11, 9, 19, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["skipped"] is False
    assert "5m" in out["closed_tfs"]
    assert (tmp_path / "tick_2026-08-11.json").exists()


def test_resolve_quotes_fills_tmpv_trent_from_store(tmp_path, monkeypatch):
    """Live batch missing TMPV/TRENT must still get usable quotes from spreads."""
    from data.ingest.store import DataStore
    from experiments import paper_live as pl
    from experiments.paper_live import _quote_usable

    db = tmp_path / "q.duckdb"
    store = DataStore(db_path=db)
    store.init_schema()
    ts = datetime.now(tz=IST)
    store.write_friction_spreads(
        pd.DataFrame(
            [
                {
                    "ts": ts,
                    "symbol": "TMPV",
                    "bid": 338.0,
                    "ask": 338.2,
                    "ltp": 338.1,
                    "half_spread_bps": 3.0,
                },
                {
                    "ts": ts,
                    "symbol": "TRENT",
                    "bid": 2995.0,
                    "ask": 2996.0,
                    "ltp": 2995.5,
                    "half_spread_bps": 1.5,
                },
                {
                    "ts": ts,
                    "symbol": "RELIANCE",
                    "bid": 1300.0,
                    "ask": 1301.0,
                    "ltp": 1300.5,
                    "half_spread_bps": 4.0,
                },
            ]
        )
    )
    store.close()

    # LTP-only live quote must NOT block store fallback (the Aug-11 failure mode)
    live = [
        Quote("RELIANCE", ltp=1300.5, bid=1300.0, ask=1301.0),
        Quote("TMPV", ltp=338.1, bid=None, ask=None),
        Quote("TRENT", ltp=2995.5, bid=None, ask=None),
    ]

    def _from_store(symbols):
        with DataStore(db_path=db) as s:
            s.init_schema()
            spreads = s.read_latest_spreads(list(symbols), valid_only=True)
        out = {}
        if spreads is None or spreads.empty:
            return out
        for r in spreads.itertuples():
            q = Quote(str(r.symbol), float(r.ltp), float(r.bid), float(r.ask))
            if _quote_usable(q):
                out[q.symbol] = q
        return out

    monkeypatch.setattr(pl, "_quotes_from_store", _from_store)

    qmap = pl._resolve_quotes(live, ["RELIANCE", "TMPV", "TRENT"])
    assert _quote_usable(qmap["RELIANCE"])
    assert _quote_usable(qmap["TMPV"])
    assert _quote_usable(qmap["TRENT"])
    assert abs(qmap["TMPV"].ask - 338.2) < 1e-9
    assert abs(qmap["TRENT"].bid - 2995.0) < 1e-9


def test_paper_live_skips_unusable_quote_without_traceback(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=30.0, vix=18.0, vix_above_median=True, expiry_week=True
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class AlwaysBuy:
        id = "A3"
        cluster = "A"
        timeframe = "1D"
        product = "CNC"

        def on_bar(self, bar, state):
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(pl, "load_enabled_strategies", lambda: {"A3": AlwaysBuy()})
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"A3": "A"})
    monkeypatch.setattr(pl, "_size_qty", lambda **kw: 10)
    # Force resolve to return LTP-only (unusable) and store refresh empty
    monkeypatch.setattr(
        pl,
        "_resolve_quotes",
        lambda quotes, symbols: {s: Quote(s, ltp=100.0, bid=None, ask=None) for s in symbols},
    )
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})

    bars = _bars(120)
    bars = bars[bars["symbol"] == "RELIANCE"].reset_index(drop=True)
    out = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=[Quote("RELIANCE", ltp=100.0, bid=None, ask=None)],
        now=datetime(2026, 8, 11, 15, 29, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["fills"] == 0
    assert any("no measured bid/ask" in e for e in (out.get("errors") or []))


def test_paper_live_buy_does_not_rebuy_when_at_target(tmp_path, monkeypatch):
    import json

    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=30.0, vix=18.0, vix_above_median=True, expiry_week=True
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class AlwaysBuy:
        id = "A3"
        cluster = "A"
        timeframe = "1D"
        product = "CNC"

        def on_bar(self, bar, state):
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(pl, "load_enabled_strategies", lambda: {"A3": AlwaysBuy()})
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"A3": "A"})
    monkeypatch.setattr(pl, "_size_qty", lambda **kw: 10)
    monkeypatch.setattr(pl, "_enrich_quotes_from_store", lambda qmap, symbols: qmap)
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2560.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})

    books = {"positions": {"A3:RELIANCE": {"qty": 10, "product": "CNC"}}, "updated_at": None}
    (tmp_path / "virtual_books.json").write_text(json.dumps(books), encoding="utf-8")

    quotes = [Quote("RELIANCE", ltp=2560.0, bid=2559.0, ask=2561.0, prev_close=2560.0)]
    bars = _bars(120)
    bars = bars[bars["symbol"] == "RELIANCE"].reset_index(drop=True)
    out = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=quotes,
        now=datetime(2026, 8, 11, 15, 29, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["fills"] == 0

    (tmp_path / "virtual_books.json").write_text(
        json.dumps({"positions": {}, "updated_at": None}), encoding="utf-8"
    )
    out2 = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=quotes,
        now=datetime(2026, 8, 11, 15, 29, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out2["fills"] >= 1

    out3 = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=quotes,
        now=datetime(2026, 8, 11, 15, 30, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert "1D" not in out3["closed_tfs"]


def test_paper_live_1d_closes_when_wall_clock_is_1530(tmp_path, monkeypatch):
    """Missed exact 15:29 stamp after ingest wait still evaluates daily strats."""
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=30.0, vix=18.0, vix_above_median=True, expiry_week=True
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 / 21 for sid in kw["strategy_ids"]},
    )
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2560.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})

    ts = pd.date_range("2026-08-13 09:15", "2026-08-13 15:29", freq="1min", tz=IST)
    bars = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["RELIANCE"] * len(ts),
            "open": 2560.0,
            "high": 2561.0,
            "low": 2559.0,
            "close": 2560.0,
            "volume": 1000.0,
        }
    )
    quotes = [Quote("RELIANCE", ltp=2560.0, bid=2559.0, ask=2561.0, prev_close=2560.0)]
    out = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=quotes,
        now=datetime(2026, 8, 13, 15, 30, 12, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["skipped"] is False
    assert "1D" in out["closed_tfs"]
