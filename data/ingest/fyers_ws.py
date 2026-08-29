from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

from data.ingest.live import record_spread_snapshot
from data.ingest.store import DataStore
from data.ingest.symbols import from_fyers_symbol, load_nifty50_symbols, to_fyers_symbol
from nse_trader.env import ENV_FILE, load_dotenv
from sim.friction.measured import Quote

logger = logging.getLogger(__name__)
IST = __import__("zoneinfo").ZoneInfo("Asia/Kolkata")

CLOSE_JOIN_SEC = 2.0


def close_fyers_bounded(fyers: Any, *, timeout_sec: float = CLOSE_JOIN_SEC) -> bool:
    """
    Fyers DataSocket.close_connection() joins ws/message/ping threads with no
    timeout. At NSE close the socket often goes half-dead and join() never
    returns, so systemd kills the oneshot before paper-live (1D) can run.

    Close in a daemon thread and continue after timeout_sec.
    Returns True if close finished in time.
    """
    done = threading.Event()

    def _run() -> None:
        try:
            fyers.close_connection()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_run, name="fyers-close", daemon=True).start()
    if done.wait(timeout_sec):
        return True
    logger.warning(
        "fyers close_connection still blocked after %.1fs — skipping join",
        timeout_sec,
    )
    return False


def _access_token(*, ensure: bool = True) -> str | None:
    load_dotenv()
    if ensure:
        try:
            from data.ingest.fyers_auth import ensure_valid_access_token

            return ensure_valid_access_token(skew_seconds=300)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fyers ensure_valid_access_token failed: %s", exc)
    from data.ingest.fyers_auth import combined_access_token

    return combined_access_token()


def _parse_quote(message: dict[str, Any]) -> Quote | None:
    """Parse Fyers SymbolUpdate into Quote.

    Prefer explicit bid_price/ask_price only — short keys ``b``/``a`` are ambiguous
    and often carry circuit bounds near the close (ask < bid).
    """
    symbol_raw = message.get("symbol") or message.get("n")
    if not symbol_raw:
        return None
    symbol = from_fyers_symbol(str(symbol_raw))
    ltp = message.get("ltp") or message.get("last_price") or message.get("lp")
    bid = message.get("bid_price")
    if bid is None:
        bid = message.get("bid")
    ask = message.get("ask_price")
    if ask is None:
        ask = message.get("ask")
    if ltp is None:
        return None
    ltp_f = float(ltp)
    try:
        bid_f = float(bid) if bid is not None else None
        ask_f = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        return None
    if bid_f is None or ask_f is None:
        return None
    if ask_f < bid_f or bid_f <= 0 or ask_f <= 0:
        return None

    def _pos(key: str, *alts: str) -> float | None:
        for k in (key, *alts):
            raw = message.get(k)
            if raw is None:
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return None

    return Quote(
        symbol=symbol,
        ltp=ltp_f,
        bid=bid_f,
        ask=ask_f,
        upper_ckt=_pos("upper_ckt", "upper_circuit", "upperPrice"),
        lower_ckt=_pos("lower_ckt", "lower_circuit", "lowerPrice"),
        prev_close=_pos("prev_close_price", "prev_close", "previous_close"),
    )


def run_fyers_websocket_ingest(
    symbols: list[str] | None = None,
    *,
    duration_sec: int = 55,
    batch_size: int = 50,
) -> dict:
    """
    Subscribe to Fyers SymbolUpdate for Nifty 50, persist spreads + 1m bars.
    Short-lived for systemd (exits after duration_sec).
    """
    access_token = _access_token(ensure=True)
    if not access_token:
        raise RuntimeError(
            f"Missing FYERS_APP_ID / FYERS_ACCESS_TOKEN in {ENV_FILE}. "
            "Use placeholder ingest or add credentials."
        )

    try:
        from fyers_apiv3.FyersWebsocket import data_ws
    except ImportError as exc:
        raise ImportError("pip install fyers-apiv3") from exc

    symbols = symbols or load_nifty50_symbols()
    fyers_symbols = [to_fyers_symbol(s) for s in symbols]

    quotes: dict[str, Quote] = {}
    last_quotes: dict[str, Quote] = {}
    bars_buffer: list[dict] = []
    stop = threading.Event()
    lock = threading.Lock()
    auth_fail = threading.Event()

    def flush() -> tuple[int, int]:
        with lock:
            qlist = list(quotes.values())
            bars = list(bars_buffer)
            quotes.clear()
            bars_buffer.clear()
        spread_rows = record_spread_snapshot(qlist) if qlist else 0
        bar_rows = 0
        if bars:
            import pandas as pd

            with DataStore() as store:
                store.init_schema()
                bar_rows = store.write_bars_1m(pd.DataFrame(bars))
        return spread_rows, bar_rows

    def onmessage(message: dict) -> None:
        if not isinstance(message, dict):
            return
        q = _parse_quote(message)
        if q is None:
            return
        with lock:
            quotes[q.symbol] = q
            last_quotes[q.symbol] = q
            bars_buffer.append(
                {
                    "ts": datetime.now(tz=IST).replace(second=0, microsecond=0),
                    "symbol": q.symbol,
                    "open": q.ltp,
                    "high": q.ltp,
                    "low": q.ltp,
                    "close": q.ltp,
                    "volume": float(message.get("vol_traded_today") or message.get("v") or 0),
                }
            )

    def onerror(msg: object) -> None:
        logger.error("fyers ws error: %s", msg)
        try:
            from data.ingest.fyers_auth import ensure_valid_access_token, is_fyers_auth_error

            if is_fyers_auth_error(str(msg)):
                auth_fail.set()
                ensure_valid_access_token(force=True)
                logger.warning("reminted Fyers token after websocket auth error")
                stop.set()
        except Exception as exc:  # noqa: BLE001
            logger.error("on-demand remint failed: %s", exc)

    def onclose(msg: object) -> None:
        logger.info("fyers ws closed: %s", msg)
        stop.set()

    def onopen() -> None:
        for i in range(0, len(fyers_symbols), batch_size):
            batch = fyers_symbols[i : i + batch_size]
            fyers.subscribe(symbols=batch, data_type="SymbolUpdate")
        logger.info("subscribed %s symbols", len(fyers_symbols))

    fyers = data_ws.FyersDataSocket(
        access_token=access_token,
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=False,
        on_connect=onopen,
        on_close=onclose,
        on_error=onerror,
        on_message=onmessage,
    )

    # connect() can block — isolate so duration_sec is honored
    def _connect() -> None:
        try:
            fyers.connect()
        except Exception as exc:  # noqa: BLE001
            logger.error("fyers connect failed: %s", exc)
            stop.set()

    threading.Thread(target=_connect, name="fyers-connect", daemon=True).start()
    deadline = time.time() + duration_sec
    total_spreads = 0
    total_bars = 0
    while time.time() < deadline and not stop.is_set():
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
        if time.time() >= deadline:
            break
        s, b = flush()
        total_spreads += s
        total_bars += b

    stop.set()
    close_fyers_bounded(fyers, timeout_sec=2.0)
    s, b = flush()
    total_spreads += s
    total_bars += b

    with lock:
        out_quotes = list(last_quotes.values())
    return {
        "mode": "fyers_websocket",
        "symbols": len(fyers_symbols),
        "duration_sec": duration_sec,
        "spread_rows": total_spreads,
        "bar_rows": total_bars,
        "auth_remint": auth_fail.is_set(),
        "quotes": out_quotes,
    }
