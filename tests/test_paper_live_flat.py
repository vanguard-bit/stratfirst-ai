"""Paper-live MIS strat flat + circuit flat."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from sim.friction.measured import Quote

IST = ZoneInfo("Asia/Kolkata")
pytestmark = pytest.mark.runtime


def _bars_to(hhmm: str) -> pd.DataFrame:
    last = pd.Timestamp(f"2026-08-14 {hhmm}", tz=IST)
    start = pd.Timestamp("2026-08-14 09:15", tz=IST)
    idx = pd.date_range(start, last, freq="1min")
    return pd.DataFrame(
        {
            "ts": idx,
            "symbol": ["RELIANCE"] * len(idx),
            "open": 2500.0,
            "high": 2501.0,
            "low": 2499.0,
            "close": 2500.0,
            "volume": 1000.0,
        }
    )


def test_mis_strat_flat_at_1515_attributed_to_strategy(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2500.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 0.0 for sid in kw["strategy_ids"]},
    )
    monkeypatch.setattr(pl, "load_enabled_strategies", lambda: {})
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"B2": "B"})

    books = {"positions": {"B2:RELIANCE": {"qty": 10, "product": "MIS"}}, "updated_at": None}
    (tmp_path / "virtual_books.json").write_text(json.dumps(books), encoding="utf-8")

    out = pl.run_paper_live_tick(
        bars_1m=_bars_to("15:15"),
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0, prev_close=2500.0)],
        now=datetime(2026, 8, 14, 15, 15, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["fills"] >= 1
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    assert int(saved["positions"]["B2:RELIANCE"]["qty"]) == 0

    # Trade parquet under live-YYYY-MM-DD when log_trades runs
    trades_dir = tmp_path / "live-2026-08-14"
    if trades_dir.exists():
        import pandas as pd

        tr = pd.read_parquet(trades_dir / "trades.parquet")
        assert (tr["strategy_id"] == "B2").any()
        assert (tr["reason"] == "MIS_STRAT_FLAT").any()


def test_mis_buy_blocked_after_strat_flat_cnc_still_enters(tmp_path, monkeypatch):
    """No new MIS longs from 15:15 (e.g. 15:19 5m close); CNC may still buy."""
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2500.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class _MisBuy:
        timeframe = "5m"
        product = "MIS"
        cluster = "B"

        def on_bar(self, bar, state):
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)

    class _CncBuy:
        timeframe = "5m"
        product = "CNC"
        cluster = "A"

        def on_bar(self, bar, state):
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(
        pl,
        "load_enabled_strategies",
        lambda: {"B2": _MisBuy(), "A3": _CncBuy()},
    )
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"B2": "B", "A3": "A"})
    (tmp_path / "virtual_books.json").write_text(
        json.dumps({"positions": {}, "updated_at": None}),
        encoding="utf-8",
    )

    pl.run_paper_live_tick(
        bars_1m=_bars_to("15:19"),
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0, prev_close=2500.0)],
        now=datetime(2026, 8, 14, 15, 19, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    pos = saved.get("positions") or {}
    assert int(pos.get("B2:RELIANCE", {}).get("qty") or 0) == 0
    assert int(pos.get("A3:RELIANCE", {}).get("qty") or 0) > 0


def test_circuit_flat_long_at_uc(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 100.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 0.0 for sid in kw["strategy_ids"]},
    )
    monkeypatch.setattr(pl, "load_enabled_strategies", lambda: {})
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"B2": "B"})

    books = {"positions": {"B2:RELIANCE": {"qty": 5, "product": "MIS"}}, "updated_at": None}
    (tmp_path / "virtual_books.json").write_text(json.dumps(books), encoding="utf-8")

    # LTP at UC — long can sell
    q = Quote(
        "RELIANCE",
        ltp=110.0,
        bid=109.9,
        ask=110.0,
        upper_ckt=110.0,
        lower_ckt=90.0,
        prev_close=100.0,
    )
    out = pl.run_paper_live_tick(
        bars_1m=_bars_to("10:00"),
        quotes=[q],
        now=datetime(2026, 8, 14, 10, 0, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    assert out["fills"] >= 1
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    assert int(saved["positions"]["B2:RELIANCE"]["qty"]) == 0


def test_mis_sell_opens_short_cnc_sell_from_flat_skips(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2500.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class _MisSell:
        timeframe = "5m"
        product = "MIS"
        cluster = "E"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)

    class _CncSell:
        timeframe = "5m"
        product = "CNC"
        cluster = "C"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(
        pl,
        "load_enabled_strategies",
        lambda: {"E1": _MisSell(), "C1": _CncSell()},
    )
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"E1": "E", "C1": "C"})
    (tmp_path / "virtual_books.json").write_text(
        json.dumps({"positions": {}, "updated_at": None}),
        encoding="utf-8",
    )
    pl.run_paper_live_tick(
        bars_1m=_bars_to("10:04"),
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0, prev_close=2500.0)],
        now=datetime(2026, 8, 14, 10, 4, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    pos = saved.get("positions") or {}
    assert int(pos.get("E1:RELIANCE", {}).get("qty") or 0) < 0
    assert int(pos.get("C1:RELIANCE", {}).get("qty") or 0) == 0


def test_mis_sell_from_flat_blocked_after_1515(tmp_path, monkeypatch):
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2500.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class _MisSell:
        timeframe = "5m"
        product = "MIS"
        cluster = "E"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(pl, "load_enabled_strategies", lambda: {"E1": _MisSell()})
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"E1": "E"})
    (tmp_path / "virtual_books.json").write_text(
        json.dumps({"positions": {}, "updated_at": None}),
        encoding="utf-8",
    )
    pl.run_paper_live_tick(
        bars_1m=_bars_to("15:19"),
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0, prev_close=2500.0)],
        now=datetime(2026, 8, 14, 15, 19, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    qty = int((saved.get("positions") or {}).get("E1:RELIANCE", {}).get("qty") or 0)
    assert qty == 0


def test_mis_sell_zero_exposure_flattens_long_does_not_open_short(tmp_path, monkeypatch):
    """A2/B1/B2-style exit: SELL + intended_exposure=0.0 must not reverse into a short."""
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    monkeypatch.setattr(pl, "_books_path", lambda: tmp_path / "virtual_books.json")
    monkeypatch.setattr(pl, "_prev_closes_from_store", lambda symbols: {"RELIANCE": 2500.0})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=20.0, vix=15.0, vix_above_median=False, expiry_week=False
        ),
    )
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 1.0 for sid in kw["strategy_ids"]},
    )

    class _ExitZero:
        timeframe = "5m"
        product = "MIS"
        cluster = "B"

        def on_bar(self, bar, state):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=0.0)

    monkeypatch.setattr(
        pl,
        "load_enabled_strategies",
        lambda: {"B2": _ExitZero(), "A2": _ExitZero()},
    )
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"B2": "B", "A2": "A"})
    (tmp_path / "virtual_books.json").write_text(
        json.dumps(
            {
                "positions": {
                    "B2:RELIANCE": {"qty": 10, "product": "MIS"},
                },
                "updated_at": None,
            }
        ),
        encoding="utf-8",
    )
    pl.run_paper_live_tick(
        bars_1m=_bars_to("10:04"),
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0, prev_close=2500.0)],
        now=datetime(2026, 8, 14, 10, 4, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "virtual_books.json").read_text(encoding="utf-8"))
    pos = saved.get("positions") or {}
    assert int(pos.get("B2:RELIANCE", {}).get("qty") or 0) == 0
    assert int(pos.get("A2:RELIANCE", {}).get("qty") or 0) == 0
