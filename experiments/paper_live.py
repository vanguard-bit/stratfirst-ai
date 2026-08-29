"""Live multi-strategy paper tick — runs after Fyers ingest."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.allocation_log import log_allocation_snapshot
from experiments.log import log_trades
from features.bar_state import closed_timeframes_for_tick, build_state, resample_symbol_tf
from meta.features import RegimeFeatures
from nse_trader.config import PortfolioConfig, ROOT, load_yaml
from sim.friction.measured import Quote
from sim.orders import OrderIntent, OrderSide, OrderType, Product
from sim.pipeline import SimPipeline
from strategies.base import Bar
from strategies.registry import load_enabled_strategies

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)


def _cluster_of() -> dict[str, str]:
    cfg = load_yaml("strategies.yaml")
    return {sid: meta["cluster"] for sid, meta in cfg["strategies"].items()}


def _meta_weights(strategies: dict, bars_1m=None, now: datetime | None = None) -> tuple[dict[str, float], RegimeFeatures]:
    port = load_yaml("portfolio.yaml")
    meta_cfg = port.get("meta_allocator", {})
    cons = meta_cfg.get("constraints", {})
    from dataclasses import replace

    from features.llm_store import load_llm_map, summarize_llm
    from meta.allocator import AllocatorConstraints
    from meta.regime import build_regime, load_or_compute_daily_weights

    constraints = AllocatorConstraints(
        max_strategy_weight=float(cons.get("max_strategy_weight", 0.25)),
        max_cluster_weight=float(cons.get("max_cluster_weight", 0.40)),
        min_cash=float(cons.get("min_cash", 0.05)),
        max_cash=float(cons.get("max_cash", 0.30)),
    )
    mode = meta_cfg.get("mode", "rules")
    regime = build_regime(bars_1m, now=now)
    llm_map = load_llm_map()
    llm_sum = summarize_llm(llm_map)
    regime = replace(
        regime,
        llm_sentiment_mean=llm_sum.mean_sentiment,
        llm_high_materiality=llm_sum.high_materiality,
        llm_as_of=llm_sum.as_of,
    )
    weights = load_or_compute_daily_weights(
        strategy_ids=list(strategies),
        cluster_of=_cluster_of(),
        regime=regime,
        mode=mode,
        constraints=constraints,
        today=(now or datetime.now(tz=IST)).date(),
    )
    try:
        log_allocation_snapshot(
            weights,
            cluster_of=_cluster_of(),
            regime={
                "adx": regime.adx,
                "vix": regime.vix,
                "expiry_week": regime.expiry_week,
                "llm_sentiment_mean": regime.llm_sentiment_mean,
                "llm_high_materiality": regime.llm_high_materiality,
                "llm_as_of": regime.llm_as_of,
            },
            mode=mode,
            ts=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("allocation log failed: %s", exc)
    return weights, regime


def _books_path() -> Path:
    raw = load_yaml("ops.yaml")
    return ROOT / raw.get("persistence", {}).get("state_dir", "data/state") / "virtual_books.json"


def _e1_session_path() -> Path:
    return _books_path().parent / "e1_session.json"


def _load_e1_session(day: str) -> set[str]:
    path = _e1_session_path()
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if str(raw.get("date") or "") != day:
        return set()
    done = raw.get("done") or []
    return {str(s) for s in done}


def _save_e1_session(day: str, done: set[str]) -> None:
    path = _e1_session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"date": day, "done": sorted(done)}, indent=2),
        encoding="utf-8",
    )


def _load_books() -> dict:
    path = _books_path()
    if not path.exists():
        return {"positions": {}, "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_books(books: dict) -> None:
    path = _books_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    books["updated_at"] = datetime.now(tz=IST).isoformat()
    path.write_text(json.dumps(books, indent=2), encoding="utf-8")


def _position_key(strategy_id: str, symbol: str) -> str:
    return f"{strategy_id}:{symbol}"


def _size_qty(
    *,
    intended_exposure: float,
    meta_weight: float,
    price: float,
    per_strategy_notional: float,
    max_single_name: float,
) -> int:
    if price <= 0 or abs(intended_exposure) == 0:
        return 0
    notional = abs(intended_exposure) * meta_weight * per_strategy_notional
    notional = min(notional, per_strategy_notional * max_single_name)
    qty = int(notional // price)
    return max(qty, 0)


def _signal_exposure(signal) -> float:
    """Keep 0.0 as flatten/no-size. Only None means 'unspecified' → 1.0."""
    exp = getattr(signal, "intended_exposure", None)
    if exp is None:
        return 1.0
    return float(exp)


def _quote_usable(q: Quote) -> bool:
    return (
        q.ltp is not None
        and float(q.ltp) > 0
        and q.bid is not None
        and q.ask is not None
        and float(q.ask) >= float(q.bid)
        and float(q.bid) > 0
        and float(q.ask) > 0
    )


def _quotes_map(quotes: list[Quote]) -> dict[str, Quote]:
    return {q.symbol: q for q in quotes if _quote_usable(q)}


def _quotes_from_store(symbols: list[str]) -> dict[str, Quote]:
    """Latest *valid* recorded spreads for symbols (ask >= bid)."""
    if not symbols:
        return {}
    try:
        from data.ingest.store import DataStore

        with DataStore() as store:
            store.init_schema()
            spreads = store.read_latest_spreads(list(symbols), valid_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spread store read failed: %s", exc)
        return {}
    if spreads is None or getattr(spreads, "empty", True):
        return {}
    out: dict[str, Quote] = {}
    for r in spreads.itertuples():
        def _opt(name: str) -> float | None:
            if not hasattr(r, name):
                return None
            v = getattr(r, name)
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if f > 0 else None

        q = Quote(
            symbol=str(r.symbol),
            ltp=float(r.ltp),
            bid=float(r.bid),
            ask=float(r.ask),
            upper_ckt=_opt("upper_ckt"),
            lower_ckt=_opt("lower_ckt"),
            prev_close=_opt("prev_close"),
        )
        if _quote_usable(q):
            out[q.symbol] = q
    return out


def _prev_closes_from_store(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    try:
        from data.ingest.store import DataStore

        with DataStore() as store:
            store.init_schema()
            return store.read_prev_closes(list(symbols))
    except Exception as exc:  # noqa: BLE001
        logger.warning("prev_close store read failed: %s", exc)
        return {}


def _merge_circuit_fields(live: Quote, stored: Quote | None) -> Quote:
    """Fill missing circuit/prev_close on live quote from store row."""
    if stored is None:
        return live
    return Quote(
        symbol=live.symbol,
        ltp=live.ltp,
        bid=live.bid,
        ask=live.ask,
        timestamp=live.timestamp,
        upper_ckt=live.upper_ckt if live.upper_ckt and live.upper_ckt > 0 else stored.upper_ckt,
        lower_ckt=live.lower_ckt if live.lower_ckt and live.lower_ckt > 0 else stored.lower_ckt,
        prev_close=live.prev_close if live.prev_close and live.prev_close > 0 else stored.prev_close,
    )


def _resolve_quotes(quotes: list[Quote], symbols: list[str]) -> dict[str, Quote]:
    """
    Build a usable quote map for fill time.

    Prefer live websocket quotes when fully usable; fill gaps from DuckDB
    friction_spreads (valid ask>=bid only). Never return LTP-only quotes —
    MeasuredFriction requires both bid and ask.
    """
    live = _quotes_map(quotes)
    store = _quotes_from_store(symbols)
    out: dict[str, Quote] = {}
    for s in symbols:
        if s in live:
            out[s] = _merge_circuit_fields(live[s], store.get(s))
        elif s in store:
            out[s] = store[s]
    # Also keep live-only names not in bars symbol list
    for s, q in live.items():
        if s not in out:
            out[s] = _merge_circuit_fields(q, store.get(s))
    return out


def _enrich_quotes_from_store(qmap: dict[str, Quote], symbols: list[str]) -> dict[str, Quote]:
    """Back-compat wrapper: fill missing keys from store."""
    missing = [s for s in symbols if s not in qmap]
    if not missing:
        return qmap
    filled = _quotes_from_store(missing)
    out = dict(qmap)
    for sym, q in filled.items():
        out.setdefault(sym, q)
    return out


def _try_flatten(
    *,
    pipeline: SimPipeline,
    quote: Quote,
    sid: str,
    symbol: str,
    qty_signed: int,
    product_name: str,
    reason: str,
    strategy_id_log: str,
    cluster: str,
    stamp: datetime,
    regime: RegimeFeatures,
    fallback_pct: float,
    prev_close: float | None,
    result: dict,
    trades: list[dict],
    positions: dict,
    key: str,
) -> bool:
    """Market-flatten one book key. Returns True if filled."""
    if qty_signed == 0 or not _quote_usable(quote):
        return False
    qty = abs(int(qty_signed))
    side = OrderSide.SELL if qty_signed > 0 else OrderSide.BUY
    product = Product.MIS if product_name == "MIS" else Product.CNC
    from sim.exchange.circuits import circuits_or_open, resolve_circuits

    uc, lc = circuits_or_open(quote, prev_close=prev_close, fallback_pct=fallback_pct)
    intent = OrderIntent(
        strategy_id=sid,
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        product=product,
    )
    try:
        pr = pipeline.process(intent, quote=quote, uc=uc, lc=lc)
    except ValueError as vex:
        msg = f"{reason} {sid}/{symbol}: {vex}"
        logger.warning(msg)
        result["errors"].append(msg)
        return False
    fill = getattr(pr, "fill", None) or getattr(pr, "fills", None)
    if pr.status != "FILLED" or fill is None:
        raw_uc, raw_lc = resolve_circuits(quote, prev_close=prev_close, fallback_pct=fallback_pct)
        msg = (
            f"{reason} {sid}/{symbol}: rejected "
            f"{getattr(pr, 'reject_reason', None) or pr.status} "
            f"(ltp={quote.ltp} uc={raw_uc} lc={raw_lc})"
        )
        logger.warning(msg)
        result["errors"].append(msg)
        return False
    trades.append(
        {
            "trade_id": str(uuid.uuid4()),
            "ts": stamp.isoformat(),
            "strategy_id": strategy_id_log,
            "cluster": cluster,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "signal_price": fill.signal_price,
            "fill_price": fill.fill_price,
            "total_cost": fill.charges.total,
            "regime_adx": regime.adx,
            "regime_vix": regime.vix,
            "expiry_flag": regime.expiry_week,
            "reason": reason,
        }
    )
    positions[key] = {"qty": 0, "product": product_name}
    result["fills"] += 1
    return True


def run_paper_live_tick(
    *,
    bars_1m,
    quotes: list[Quote],
    now: datetime | None = None,
    mode: str = "fyers_websocket",
    out_dir: Path | None = None,
) -> dict:
    """
    Evaluate all enabled strategies on TF bars that just closed.
    Skips entirely when mode is placeholder (no fake fills).
    """
    stamp = now or datetime.now(tz=IST)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=IST)

    result = {
        "ts": stamp.isoformat(),
        "mode": mode,
        "skipped": False,
        "closed_tfs": [],
        "signals": 0,
        "fills": 0,
        "errors": [],
        "n_strategies": 0,
    }

    if mode == "placeholder":
        result["skipped"] = True
        result["reason"] = "placeholder_ingest"
        return result

    if bars_1m is None or getattr(bars_1m, "empty", True):
        result["skipped"] = True
        result["reason"] = "no_bars"
        return result

    import pandas as pd

    bars_1m = bars_1m.copy()
    bars_1m["ts"] = pd.to_datetime(bars_1m["ts"])

    tfs = closed_timeframes_for_tick(stamp, bars_1m)
    result["closed_tfs"] = tfs

    strategies = load_enabled_strategies()
    result["n_strategies"] = len(strategies)
    weights, regime = _meta_weights(strategies, bars_1m=bars_1m, now=stamp)
    result["regime"] = {
        "adx": regime.adx,
        "vix": regime.vix,
        "vix_above_median": regime.vix_above_median,
        "expiry_week": regime.expiry_week,
        "nifty_return_20d": regime.nifty_return_20d,
        "llm_sentiment_mean": regime.llm_sentiment_mean,
        "llm_high_materiality": regime.llm_high_materiality,
        "llm_as_of": regime.llm_as_of,
    }
    cluster_of = _cluster_of()

    from features.llm_store import inject_symbol_llm, load_llm_map

    llm_map = load_llm_map()
    cfg = PortfolioConfig.load()
    port = load_yaml("portfolio.yaml")
    per_notional = float(port.get("virtual_books", {}).get("per_strategy_notional", 1_000_000))
    max_single = float(port.get("position_sizing", {}).get("max_single_name", 0.20))
    flat_time = str(port.get("risk", {}).get("intraday_flat_time", "15:20"))
    strat_flat_time = str(port.get("risk", {}).get("mis_strat_flat_time", "15:15"))
    circuit_pct = float(port.get("risk", {}).get("circuit_fallback_pct", 0.10))

    registry = cfg.fees_registry_path
    if not registry.exists():
        result["skipped"] = True
        result["reason"] = "no_fee_registry"
        return result

    pipeline = SimPipeline(registry_path=registry, square_off_time=flat_time)
    books = _load_books()
    positions: dict = books.setdefault("positions", {})

    trades: list[dict] = []
    base = out_dir or (cfg.store_path / "experiments" / "live")
    day = stamp.date().isoformat()
    e1_done = _load_e1_session(day)
    # Any open E1 position counts as already used for that symbol today.
    for key, pos in positions.items():
        if key.startswith("E1:") and int(pos.get("qty", 0)) != 0:
            e1_done.add(key.split(":", 1)[1])
    e1_dirty = False

    symbols = sorted(set(bars_1m["symbol"].astype(str)))
    # Include open book symbols so circuit/MIS flat can resolve quotes
    for key in positions:
        if ":" in key and int(positions[key].get("qty", 0)) != 0:
            symbols.append(key.split(":", 1)[1])
    symbols = sorted(set(symbols))
    qmap = _resolve_quotes(quotes, symbols)
    prev_closes = _prev_closes_from_store(symbols)

    from sim.exchange.circuits import at_circuit, circuits_or_open

    hhmm = stamp.strftime("%H:%M")

    def _pc_for(symbol: str, quote: Quote) -> float | None:
        if quote.prev_close and float(quote.prev_close) > 0:
            return float(quote.prev_close)
        return prev_closes.get(symbol)

    # Circuit-hit square-off (MIS + CNC) — before new entries
    for key, pos in list(positions.items()):
        qty_signed = int(pos.get("qty", 0))
        if qty_signed == 0:
            continue
        sid, symbol = key.split(":", 1)
        quote = qmap.get(symbol)
        if quote is None or not _quote_usable(quote):
            continue
        pc = _pc_for(symbol, quote)
        # Force flat only on exchange-reported bands (not ±10% fallback),
        # so a stale bars_1d prev_close cannot liquidate the book.
        fy_uc, fy_lc = quote.upper_ckt, quote.lower_ckt
        if not (
            fy_uc is not None
            and fy_lc is not None
            and float(fy_uc) > 0
            and float(fy_lc) > 0
            and float(fy_uc) >= float(fy_lc)
        ):
            continue
        if not at_circuit(float(quote.ltp), float(fy_uc), float(fy_lc)):
            continue
        _try_flatten(
            pipeline=pipeline,
            quote=quote,
            sid=sid,
            symbol=symbol,
            qty_signed=qty_signed,
            product_name=str(pos.get("product", "CNC")),
            reason="CIRCUIT_FLAT",
            strategy_id_log=sid,
            cluster=cluster_of.get(sid, "G"),
            stamp=stamp,
            regime=regime,
            fallback_pct=circuit_pct,
            prev_close=pc,
            result=result,
            trades=trades,
            positions=positions,
            key=key,
        )

    # Strat-owned MIS flat [15:15, 15:20)
    if hhmm >= strat_flat_time and hhmm < flat_time:
        for key, pos in list(positions.items()):
            if pos.get("product") != "MIS" or int(pos.get("qty", 0)) == 0:
                continue
            sid, symbol = key.split(":", 1)
            quote = qmap.get(symbol)
            if quote is None:
                continue
            _try_flatten(
                pipeline=pipeline,
                quote=quote,
                sid=sid,
                symbol=symbol,
                qty_signed=int(pos["qty"]),
                product_name="MIS",
                reason="MIS_STRAT_FLAT",
                strategy_id_log=sid,
                cluster=cluster_of.get(sid, "G"),
                stamp=stamp,
                regime=regime,
                fallback_pct=circuit_pct,
                prev_close=_pc_for(symbol, quote),
                result=result,
                trades=trades,
                positions=positions,
                key=key,
            )

    # BROKER safety net ≥ 15:20 for leftover MIS
    if hhmm >= flat_time:
        for key, pos in list(positions.items()):
            if pos.get("product") != "MIS" or int(pos.get("qty", 0)) == 0:
                continue
            sid, symbol = key.split(":", 1)
            quote = qmap.get(symbol)
            if quote is None:
                continue
            _try_flatten(
                pipeline=pipeline,
                quote=quote,
                sid=sid,
                symbol=symbol,
                qty_signed=int(pos["qty"]),
                product_name="MIS",
                reason="MIS_SQUARE_OFF",
                strategy_id_log="BROKER",
                cluster=cluster_of.get(sid, "G"),
                stamp=stamp,
                regime=regime,
                fallback_pct=circuit_pct,
                prev_close=_pc_for(symbol, quote),
                result=result,
                trades=trades,
                positions=positions,
                key=key,
            )

    # Cache resampled frames per TF
    tf_frames: dict[str, object] = {}
    for tf in tfs:
        if tf == "1m":
            continue  # strategies use 5m+ mostly; 1m strategies rare
        try:
            tf_frames[tf] = resample_symbol_tf(bars_1m, tf)
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"resample {tf}: {exc}")

    for sid, strat in strategies.items():
        tf = getattr(strat, "timeframe", "1D")
        if tf not in tfs and tf not in tf_frames:
            # daily/weekly only on their close
            if not (tf in tfs):
                continue
        if tf == "1m":
            frame = bars_1m
        else:
            if tf not in tfs:
                continue
            frame = tf_frames.get(tf)
            if frame is None or getattr(frame, "empty", True):
                continue

        product_name = getattr(strat, "product", "CNC")
        product = Product.MIS if product_name == "MIS" else Product.CNC
        meta_w = float(weights.get(sid, 0.0))

        for symbol in symbols:
            quote = qmap.get(symbol)
            if quote is None or quote.ltp is None:
                continue
            try:
                state = build_state(
                    frame,
                    symbol,
                    timeframe=tf,
                    universe_bars=frame,
                    now=stamp,
                )
                if not state or symbol not in set(frame["symbol"].astype(str)):
                    continue
                pos = positions.get(_position_key(sid, symbol), {})
                cur_qty_state = int(pos.get("qty", 0))
                state["in_position"] = cur_qty_state != 0
                state["position_qty"] = cur_qty_state
                state["e1_traded_today"] = symbol in e1_done if sid == "E1" else False
                state["session_hhmm"] = hhmm
                state["adx"] = regime.adx
                state["vix_above_median"] = regime.vix_above_median
                state["expiry_week"] = regime.expiry_week
                inject_symbol_llm(state, symbol, llm_map)

                last = frame[frame["symbol"] == symbol].sort_values("ts").iloc[-1]
                bar = Bar(
                    ts=str(last["ts"]),
                    symbol=symbol,
                    open=float(last["open"]),
                    high=float(last["high"]),
                    low=float(last["low"]),
                    close=float(last["close"]),
                    volume=float(last.get("volume", 0.0)),
                    timeframe=tf,
                )
                signal = strat.on_bar(bar, state)
                result["signals"] += 1

                action = (signal.action or "HOLD").upper()
                if action == "HOLD":
                    continue

                key = _position_key(sid, symbol)
                cur_qty = int(positions.get(key, {}).get("qty", 0))
                # After 15:15: no *new* MIS risk. Cover leftover shorts / flatten leftover
                # longs still allowed (15:15 flatten-first already handles most of this).
                if product_name == "MIS" and hhmm >= strat_flat_time:
                    if action == "BUY" and cur_qty >= 0:
                        continue
                    if action == "SELL" and cur_qty <= 0:
                        continue
                    if action in ("BUY", "SELL"):
                        action = "FLAT"

                exp = _signal_exposure(signal)
                # A2/B1/B2 emit SELL + exposure 0.0 to *exit*, not to open a short.
                if (
                    action == "SELL"
                    and str(product_name).upper() == "MIS"
                    and abs(exp) == 0.0
                ):
                    if cur_qty == 0:
                        continue
                    action = "FLAT"

                # Re-resolve if live quote went stale/partial (TMPV/TRENT class failures)
                if not _quote_usable(quote):
                    refreshed = _quotes_from_store([symbol]).get(symbol)
                    if refreshed is not None:
                        quote = refreshed
                        qmap[symbol] = refreshed
                if not _quote_usable(quote):
                    msg = f"{sid}/{symbol}: skip fill — no measured bid/ask"
                    logger.warning(msg)
                    result["errors"].append(msg)
                    continue

                # Bake-off measure ledger (unit notional, ignore meta weight).
                # Isolated: never fails capital path.
                try:
                    from experiments.measure_ledger import (
                        MEASURE_BOOKS,
                        MEASURE_FILLS,
                        process_measure_signal,
                    )

                    mrow = process_measure_signal(
                        pipeline=pipeline,
                        quote=quote,
                        strategy_id=sid,
                        cluster=cluster_of.get(sid, "?"),
                        action=action,
                        product_name=product_name,
                        ts=stamp,
                        books_path=MEASURE_BOOKS,
                        fills_path=MEASURE_FILLS,
                    )
                    if mrow is not None:
                        result["measure_fills"] = int(result.get("measure_fills", 0)) + 1
                except Exception as mex:  # noqa: BLE001
                    logger.warning("measure ledger skipped %s/%s: %s", sid, symbol, mex)

                if action == "FLAT":
                    if cur_qty == 0:
                        continue
                    target = 0
                elif action == "BUY":
                    size = _size_qty(
                        intended_exposure=exp,
                        meta_weight=meta_w,
                        price=float(quote.ltp),
                        per_strategy_notional=per_notional,
                        max_single_name=max_single,
                    )
                    if size <= 0:
                        continue
                    target = size
                elif action == "SELL":
                    size = _size_qty(
                        intended_exposure=exp,
                        meta_weight=meta_w,
                        price=float(quote.ltp),
                        per_strategy_notional=per_notional,
                        max_single_name=max_single,
                    )
                    if str(product_name).upper() != "MIS":
                        if cur_qty <= 0:
                            continue
                        target = 0
                    elif size <= 0:
                        if cur_qty > 0:
                            target = 0
                        else:
                            continue
                    else:
                        target = -size
                else:
                    continue

                delta = target - cur_qty
                if delta == 0:
                    continue
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                qty = abs(delta)

                intent = OrderIntent(
                    strategy_id=sid,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    product=product,
                )
                uc, lc = circuits_or_open(
                    quote,
                    prev_close=_pc_for(symbol, quote),
                    fallback_pct=circuit_pct,
                )
                try:
                    pr = pipeline.process(intent, quote=quote, uc=uc, lc=lc)
                except ValueError as vex:
                    # MeasuredFriction refuse — skip symbol, do not kill the strat loop
                    msg = f"{sid}/{symbol}: {vex}"
                    logger.warning(msg)
                    result["errors"].append(msg)
                    continue
                fill = getattr(pr, "fill", None) or getattr(pr, "fills", None)
                if pr.status != "FILLED" or fill is None:
                    continue

                signed = qty if side == OrderSide.BUY else -qty
                new_qty = cur_qty + signed
                positions[key] = {"qty": new_qty, "product": product_name}
                if (
                    sid == "E1"
                    and cur_qty == 0
                    and new_qty != 0
                    and symbol not in e1_done
                ):
                    e1_done.add(symbol)
                    e1_dirty = True
                trades.append(
                    {
                        "trade_id": str(uuid.uuid4()),
                        "ts": stamp.isoformat(),
                        "strategy_id": sid,
                        "cluster": cluster_of.get(sid, "?"),
                        "symbol": symbol,
                        "side": side.value,
                        "qty": qty,
                        "signal_price": fill.signal_price,
                        "fill_price": fill.fill_price,
                        "total_cost": fill.charges.total,
                        "regime_adx": regime.adx,
                        "regime_vix": regime.vix,
                        "expiry_flag": regime.expiry_week,
                    }
                )
                result["fills"] += 1
            except Exception as exc:  # noqa: BLE001
                msg = f"{sid}/{symbol}: {exc}"
                logger.exception("strategy tick failed: %s", msg)
                result["errors"].append(msg)

    if e1_dirty or e1_done:
        _save_e1_session(day, e1_done)

    books["positions"] = positions
    _save_books(books)

    if trades:
        log_trades(trades, f"live-{day}", base_dir=base)

    tick_path = base / f"tick_{day}.json"
    base.mkdir(parents=True, exist_ok=True)
    # append-friendly summary: overwrite last tick snapshot
    tick_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["path"] = str(tick_path)
    return result
