"""Measure ledger: friction parity, meta-weight ignore, capital isolation."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from sim.friction.measured import Quote
from sim.orders import OrderIntent, OrderSide, OrderType, Product
from sim.pipeline import SimPipeline

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def fee_registry(tmp_path):
    # Use project registry if present; else skip
    from nse_trader.config import PortfolioConfig

    cfg = PortfolioConfig.load()
    path = cfg.fees_registry_path
    if not path.exists():
        pytest.skip("fee registry missing")
    return path


def test_measure_uses_same_pipeline_fill(fee_registry, tmp_path):
    from experiments.measure_ledger import process_measure_signal, unit_qty

    pipe = SimPipeline(registry_path=fee_registry)
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)
    qty = unit_qty(2500.0, notional=10_000)
    intent = OrderIntent(
        strategy_id="CAP",
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
        product=Product.CNC,
    )
    capital = pipe.process(intent, quote=quote, uc=3000, lc=2000)
    fill_c = capital.fill
    assert capital.status == "FILLED" and fill_c is not None

    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    row = process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="A1",
        cluster="A",
        action="BUY",
        product_name="CNC",
        ts=datetime(2026, 8, 11, 10, 0, tzinfo=IST),
        books_path=books,
        fills_path=fills,
        unit_notional=10_000,
    )
    assert row is not None
    assert row["qty"] == qty
    assert abs(row["fill_price"] - fill_c.fill_price) < 1e-9
    assert abs(row["total_cost"] - fill_c.charges.total) < 1e-9
    assert row["meta_weight_ignored"] is True


def test_measure_fill_with_zero_meta_does_not_touch_capital_books(fee_registry, tmp_path, monkeypatch):
    from experiments import measure_ledger as ml
    from experiments import paper_live as pl
    from meta.features import RegimeFeatures
    from strategies.base import Signal

    capital_books = tmp_path / "virtual_books.json"
    capital_books.write_text('{"positions": {}}')
    monkeypatch.setattr(pl, "_books_path", lambda: capital_books)
    monkeypatch.setattr(
        ml,
        "MEASURE_BOOKS",
        tmp_path / "measure_books.json",
    )
    monkeypatch.setattr(ml, "MEASURE_FILLS", tmp_path / "measure_fills.parquet")

    monkeypatch.setattr(
        "meta.regime.build_regime",
        lambda bars_1m=None, now=None: RegimeFeatures(
            adx=30.0, vix=18.0, vix_above_median=False, expiry_week=False
        ),
    )
    # All meta weights zero → capital BUY skipped; measure should still fill
    monkeypatch.setattr(
        "meta.regime.load_or_compute_daily_weights",
        lambda **kw: {sid: 0.0 for sid in kw["strategy_ids"]},
    )

    class _Stub:
        timeframe = "5m"
        product = "CNC"
        cluster = "A"

        def on_bar(self, bar, state):
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)

    monkeypatch.setattr(
        pl,
        "load_enabled_strategies",
        lambda: {"A1": _Stub()},
    )
    monkeypatch.setattr(pl, "_cluster_of", lambda: {"A1": "A"})
    monkeypatch.setattr(pl, "_quotes_from_store", lambda symbols: {})
    monkeypatch.setattr(pl, "_enrich_quotes_from_store", lambda qmap, symbols: qmap)

    import pandas as pd

    ts = pd.date_range("2026-08-11 09:15", periods=30, freq="1min", tz=IST)
    bars = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "RELIANCE",
            "open": 2500.0,
            "high": 2501.0,
            "low": 2499.0,
            "close": 2500.0,
            "volume": 1000.0,
        }
    )
    out = pl.run_paper_live_tick(
        bars_1m=bars,
        quotes=[Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)],
        now=datetime(2026, 8, 11, 9, 19, tzinfo=IST),
        mode="fyers_websocket",
        out_dir=tmp_path / "live",
    )
    assert out.get("fills", 0) == 0  # capital blocked by meta_w=0
    assert out.get("measure_fills", 0) >= 1
    import json

    cap = json.loads(capital_books.read_text())
    assert cap.get("positions") == {} or all(
        int(p.get("qty", 0)) == 0 for p in (cap.get("positions") or {}).values()
    )


def test_aggregate_opening_day_shows_friction_cost(fee_registry, tmp_path):
    from experiments.measure_ledger import (
        UNIT_NOTIONAL,
        aggregate_strat_day,
        process_measure_signal,
    )

    pipe = SimPipeline(registry_path=fee_registry)
    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    out = tmp_path / "strat.parquet"
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)
    process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="A1",
        cluster="A",
        action="BUY",
        product_name="CNC",
        ts=datetime(2026, 8, 11, 10, 0, tzinfo=IST),
        books_path=books,
        fills_path=fills,
    )
    day = aggregate_strat_day(
        day="2026-08-11",
        marks={"RELIANCE": 2500.0},
        books_path=books,
        fills_path=fills,
        out_path=out,
    )
    assert len(day) == 1
    pnl = float(day.iloc[0]["pnl"])
    ret = float(day.iloc[0]["ret"])
    deployed = float(day.iloc[0]["deployed"])
    # Opening buy marked at same px → pnl ≈ -fees < 0
    assert pnl < 0
    assert ret < 0
    assert abs(ret - pnl / deployed) < 1e-12
    # Single-name book: denom ≈ one unit
    assert abs(deployed - UNIT_NOTIONAL) < 2_000


def test_aggregate_does_not_book_opening_inventory_as_pnl(tmp_path):
    """Positions from a prior fill day with no snapshot must not print as +88%."""
    import json

    import pandas as pd

    from experiments.measure_ledger import aggregate_strat_day

    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    books.write_text(
        json.dumps(
            {
                "positions": {
                    "E1:RELIANCE": {"qty": 3, "product": "MIS", "avg_px": 2561.0},
                    "E1:TCS": {"qty": 2, "product": "MIS", "avg_px": 3561.0},
                },
                "cash_pnl": {},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        columns=["date", "strategy_id", "cluster", "symbol", "side", "qty", "fill_price", "total_cost"]
    ).to_parquet(fills, index=False)

    day = aggregate_strat_day(
        day="2026-08-12",
        marks={"RELIANCE": 2561.0, "TCS": 3561.0},
        books_path=books,
        fills_path=fills,
        out_path=tmp_path / "strat.parquet",
    )
    assert len(day) == 1
    assert abs(float(day.iloc[0]["pnl"])) < 1.0
    assert abs(float(day.iloc[0]["ret"])) < 0.01


def test_aggregate_ignores_marks_far_from_fill_px(tmp_path):
    """Placeholder fill px vs real NSE close must not print as −40%."""
    import json

    import pandas as pd

    from experiments.measure_ledger import aggregate_strat_day

    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    books.write_text(
        json.dumps(
            {
                "positions": {
                    "A3:RELIANCE": {"qty": 3, "product": "CNC", "avg_px": 2561.0},
                },
                "cash_pnl": {},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        columns=["date", "strategy_id", "cluster", "symbol", "side", "qty", "fill_price", "total_cost"]
    ).to_parquet(fills, index=False)
    day = aggregate_strat_day(
        day="2026-08-12",
        marks={"RELIANCE": 1324.0},
        books_path=books,
        fills_path=fills,
        out_path=tmp_path / "strat.parquet",
    )
    assert abs(float(day.iloc[0]["pnl"])) < 1.0
    assert abs(float(day.iloc[0]["mtm"]) - 3 * 2561.0) < 1.0


def test_aggregate_ret_scales_with_names_deployed(fee_registry, tmp_path):
    from experiments.measure_ledger import aggregate_strat_day, process_measure_signal

    pipe = SimPipeline(registry_path=fee_registry)
    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    quotes = [
        Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0),
        Quote("TCS", ltp=3500.0, bid=3499.0, ask=3501.0),
    ]
    for q in quotes:
        process_measure_signal(
            pipeline=pipe,
            quote=q,
            strategy_id="B2",
            cluster="B",
            action="BUY",
            product_name="MIS",
            ts=datetime(2026, 8, 14, 10, 0, tzinfo=IST),
            books_path=books,
            fills_path=fills,
        )
    day = aggregate_strat_day(
        day="2026-08-14",
        marks={"RELIANCE": 2500.0, "TCS": 3500.0},
        books_path=books,
        fills_path=fills,
        out_path=tmp_path / "strat.parquet",
    )
    pnl = float(day.iloc[0]["pnl"])
    ret = float(day.iloc[0]["ret"])
    deployed = float(day.iloc[0]["deployed"])
    assert pnl < 0
    assert deployed > 15_000  # two names, not one 10k unit
    assert abs(ret - pnl / deployed) < 1e-12
    assert abs(ret) < abs(pnl) / 10_000 - 1e-9  # smaller than old 1-unit scale


def test_rebuild_skips_inventory_as_profit_on_gap_day(fee_registry, tmp_path):
    from experiments.measure_ledger import (
        process_measure_signal,
        rebuild_strat_daily_from_fills,
    )

    pipe = SimPipeline(registry_path=fee_registry)
    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    out = tmp_path / "strat.parquet"
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)
    process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="E1",
        cluster="E",
        action="BUY",
        product_name="MIS",
        ts=datetime(2026, 8, 11, 10, 0, tzinfo=IST),
        books_path=books,
        fills_path=fills,
    )
    rebuilt = rebuild_strat_daily_from_fills(
        days=["2026-08-11", "2026-08-12"],
        fills_path=fills,
        books_path=books,
        out_path=out,
        marks_by_day={
            "2026-08-11": {"RELIANCE": 2500.0},
            "2026-08-12": {"RELIANCE": 2500.0},
        },
    )
    gap = rebuilt[rebuilt["date"].astype(str) == "2026-08-12"]
    assert len(gap) == 1
    assert abs(float(gap.iloc[0]["pnl"])) < 1.0
    assert abs(float(gap.iloc[0]["ret"])) < 0.01


def test_valid_fills_drop_duplicates_and_orphan_sells():
    import pandas as pd

    from experiments.measure_ledger import valid_measure_fills

    fills = pd.DataFrame(
        [
            {
                "ts": "2026-08-11T09:19:00+05:30",
                "date": "2026-08-11",
                "strategy_id": "E1",
                "symbol": "RELIANCE",
                "side": "BUY",
                "qty": 3,
                "fill_price": 2561.0,
                "total_cost": 1.0,
            },
            {
                "ts": "2026-08-11T09:19:00+05:30",
                "date": "2026-08-11",
                "strategy_id": "E1",
                "symbol": "RELIANCE",
                "side": "BUY",
                "qty": 3,
                "fill_price": 2561.0,
                "total_cost": 1.0,
            },
            {
                "ts": "2026-08-11T09:19:00+05:30",
                "date": "2026-08-11",
                "strategy_id": "E2",
                "symbol": "RELIANCE",
                "side": "SELL",
                "qty": 7,
                "fill_price": 2559.0,
                "total_cost": 1.0,
            },
        ]
    )
    out = valid_measure_fills(fills)
    assert len(out) == 1
    assert str(out.iloc[0]["strategy_id"]) == "E1"


def test_measure_mis_sell_opens_short_and_buy_covers(fee_registry, tmp_path):
    from experiments.measure_ledger import load_measure_books, process_measure_signal

    pipe = SimPipeline(registry_path=fee_registry)
    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)
    sell = process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="E1",
        cluster="E",
        action="SELL",
        product_name="MIS",
        ts=datetime(2026, 8, 17, 10, 0, tzinfo=IST),
        books_path=books,
        fills_path=fills,
    )
    assert sell is not None
    pos = load_measure_books(books)["positions"]["E1:RELIANCE"]
    assert int(pos["qty"]) < 0
    buy = process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="E1",
        cluster="E",
        action="BUY",
        product_name="MIS",
        ts=datetime(2026, 8, 17, 10, 5, tzinfo=IST),
        books_path=books,
        fills_path=fills,
    )
    assert buy is not None
    pos2 = load_measure_books(books)["positions"]["E1:RELIANCE"]
    assert int(pos2["qty"]) > 0


def test_measure_cnc_sell_from_flat_is_noop(fee_registry, tmp_path):
    from experiments.measure_ledger import load_measure_books, process_measure_signal

    pipe = SimPipeline(registry_path=fee_registry)
    books = tmp_path / "books.json"
    fills = tmp_path / "fills.parquet"
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.0, ask=2501.0)
    row = process_measure_signal(
        pipeline=pipe,
        quote=quote,
        strategy_id="C1",
        cluster="C",
        action="SELL",
        product_name="CNC",
        ts=datetime(2026, 8, 17, 10, 0, tzinfo=IST),
        books_path=books,
        fills_path=fills,
    )
    assert row is None
    books_d = load_measure_books(books)
    qty = int((books_d.get("positions") or {}).get("C1:RELIANCE", {}).get("qty") or 0)
    assert qty == 0

