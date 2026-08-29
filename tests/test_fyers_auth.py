"""Fyers auth helpers + on-demand remint."""

from __future__ import annotations

import base64
import json
import time

from nse_trader.env_file import extract_auth_code, looks_like_access_token


def test_looks_like_access_token():
    assert looks_like_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb")
    assert looks_like_access_token("APP-100:eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb")
    assert not looks_like_access_token("BQWZDVAUXX")
    assert not looks_like_access_token("")


def test_extract_auth_code_from_url():
    url = "https://trade.fyers.in/api-login/redirect-uri/index.html?auth_code=abc%2B123&state=nse"
    assert extract_auth_code(url) == "abc+123"
    assert extract_auth_code("plaincode") == "plaincode"


def _fake_jwt(*, exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_is_fyers_auth_error():
    from data.ingest.fyers_auth import is_fyers_auth_error

    assert is_fyers_auth_error({"s": "error", "code": -15, "message": "Please provide valid token"})
    assert is_fyers_auth_error("Token is expired")
    assert not is_fyers_auth_error(
        {
            "s": "error",
            "code": -16,
            "message": "Refresh token API is currently disabled to comply with SEBI regulations.",
        }
    )
    assert not is_fyers_auth_error({"s": "ok", "code": 200})


def test_ensure_remints_when_expired(monkeypatch):
    from data.ingest import fyers_auth as fa

    expired = _fake_jwt(exp=int(time.time()) - 10)
    fresh = _fake_jwt(exp=int(time.time()) + 3600)
    state = {"tok": f"APP-100:{expired}", "calls": 0}

    monkeypatch.setenv("FYERS_APP_ID", "APP-100")
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", state["tok"])

    def fake_load(_path=None):
        import os

        os.environ["FYERS_ACCESS_TOKEN"] = state["tok"]
        os.environ["FYERS_APP_ID"] = "APP-100"

    def fake_remint():
        state["calls"] += 1
        state["tok"] = f"APP-100:{fresh}"
        import os

        os.environ["FYERS_ACCESS_TOKEN"] = state["tok"]
        return {"ok": True, "method": "totp", "access_token_len": len(fresh)}

    monkeypatch.setattr(fa, "load_dotenv", fake_load)
    out = fa.ensure_valid_access_token(skew_seconds=60, remint_fn=fake_remint)
    assert state["calls"] == 1
    assert out.endswith(fresh)
    # Second call should not remint (still fresh)
    out2 = fa.ensure_valid_access_token(skew_seconds=60, remint_fn=fake_remint)
    assert state["calls"] == 1
    assert out2 == out
