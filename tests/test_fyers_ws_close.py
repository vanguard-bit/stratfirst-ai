"""Fyers websocket close must not block the ingest oneshot."""

from __future__ import annotations

import time


def test_close_fyers_bounded_returns_when_sdk_join_hangs():
    from data.ingest.fyers_ws import close_fyers_bounded

    class _Hang:
        def close_connection(self) -> None:
            time.sleep(30)

    t0 = time.monotonic()
    ok = close_fyers_bounded(_Hang(), timeout_sec=0.3)
    elapsed = time.monotonic() - t0
    assert ok is False
    assert elapsed < 2.0


def test_close_fyers_bounded_true_when_close_returns():
    from data.ingest.fyers_ws import close_fyers_bounded

    class _Ok:
        def close_connection(self) -> None:
            return None

    assert close_fyers_bounded(_Ok(), timeout_sec=1.0) is True
