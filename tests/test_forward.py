"""Fyers ingest + forward validation tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.runtime


def test_fyers_symbol_mapping():
    from data.ingest.symbols import from_fyers_symbol, to_fyers_symbol

    assert to_fyers_symbol("RELIANCE") == "NSE:RELIANCE-EQ"
    assert from_fyers_symbol("NSE:TCS-EQ") == "TCS"


def test_ingest_fallback_without_fyers_creds(monkeypatch, tmp_path):
    """Fallback must work even when a real project `.env` exists on disk.

    Fyers helpers call ``load_dotenv()``, which would remint credentials from the
    real env file unless we point ENV_FILE at an empty temp path and clear FYERS_*.
    """
    import os

    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setattr("nse_trader.env.ENV_FILE", empty_env)
    monkeypatch.setattr("data.ingest.fyers_auth.ENV_FILE", empty_env)
    monkeypatch.setattr("data.ingest.fyers_ws.ENV_FILE", empty_env)
    for key in list(os.environ):
        if key.startswith("FYERS_"):
            monkeypatch.delenv(key, raising=False)

    # Never remint / open a websocket even if something leaks credentials.
    monkeypatch.setattr(
        "data.ingest.fyers_auth.ensure_valid_access_token",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "data.ingest.fyers_auth.combined_access_token",
        lambda: None,
    )

    from data.ingest.live import run_ingest_once

    result = run_ingest_once(["RELIANCE"], duration_sec=1)
    assert result["mode"] == "placeholder"
    assert result["spread_rows"] >= 1


def test_forward_validation_one_day(tmp_path):
    from experiments.forward import run_forward_validation

    manifest = run_forward_validation(days=1, out_dir=tmp_path)
    assert manifest["days"] == 1
    assert manifest["health_last_ok"] is True
    assert (tmp_path / "forward_manifest.json").exists()
