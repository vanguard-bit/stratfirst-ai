"""Fyers OAuth: browser login, refresh (if enabled), or TOTP auto-login."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import requests

from nse_trader.env import ENV_FILE, load_dotenv, require_env
from nse_trader.env_file import (
    extract_auth_code,
    looks_like_access_token,
    migrate_secret_misplaced_as_access_token,
    upsert_env,
)

logger = logging.getLogger(__name__)

DEFAULT_REDIRECT = "https://trade.fyers.in/api-login/redirect-uri/index.html"
REFRESH_URL = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
VAGATOR = "https://api-t2.fyers.in/vagator/v2"
TOKEN_URL = "https://api-t1.fyers.in/api/v3/token"

_remint_lock = threading.Lock()


def _creds() -> tuple[str, str, str]:
    load_dotenv()
    migrate_secret_misplaced_as_access_token()
    app_id = require_env("FYERS_APP_ID")
    secret = os.environ.get("FYERS_SECRET_KEY", "").strip()
    if not secret:
        raise RuntimeError(
            f"Missing FYERS_SECRET_KEY in {ENV_FILE}. "
            "Put the dashboard Secret ID there (not in FYERS_ACCESS_TOKEN)."
        )
    redirect = os.environ.get("FYERS_REDIRECT_URI", DEFAULT_REDIRECT).strip() or DEFAULT_REDIRECT
    return app_id, secret, redirect


def _app_id_hash(app_id: str, secret: str) -> str:
    return hashlib.sha256(f"{app_id}:{secret}".encode()).hexdigest()


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _session():
    from fyers_apiv3 import fyersModel

    app_id, secret, redirect = _creds()
    return (
        app_id,
        secret,
        fyersModel.SessionModel(
            client_id=app_id,
            secret_key=secret,
            redirect_uri=redirect,
            response_type="code",
            grant_type="authorization_code",
            state="nse-trader",
        ),
    )


def jwt_payload(token: str) -> dict[str, Any] | None:
    """Decode JWT payload without verifying signature (expiry checks only)."""
    raw = token.strip()
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    try:
        pad = "=" * ((4 - len(parts[1]) % 4) % 4)
        import json

        return json.loads(base64.urlsafe_b64decode(parts[1] + pad))
    except Exception:  # noqa: BLE001
        return None


def token_seconds_left(token: str | None = None) -> float | None:
    load_dotenv()
    tok = (token or os.environ.get("FYERS_ACCESS_TOKEN", "")).strip()
    if not tok:
        return None
    payload = jwt_payload(tok)
    if not payload or "exp" not in payload:
        return None
    return float(payload["exp"]) - time.time()


def is_fyers_auth_error(resp: Any) -> bool:
    """True if a Fyers API/WS response indicates bad/expired token."""
    if resp is None:
        return False
    if isinstance(resp, str):
        low = resp.lower()
        return any(
            s in low
            for s in (
                "token is expired",
                "valid token",
                "authenticate",
                "unauthorized",
                "invalid token",
            )
        )
    if not isinstance(resp, dict):
        return False
    code = resp.get("code")
    if code in (-15, -16, -17, -209):
        # -16 is also used for SEBI refresh disable; only treat as auth when message says so
        msg = str(resp.get("message", "")).lower()
        if code == -16 and "disabled" in msg and "refresh" in msg:
            return False
        if code in (-15, -17, -209):
            return True
        return any(s in msg for s in ("authenticate", "token", "unauthorized"))
    if resp.get("s") == "error":
        msg = str(resp.get("message", "")).lower()
        return any(
            s in msg
            for s in (
                "token is expired",
                "valid token",
                "authenticate",
                "unauthorized",
                "invalid token",
            )
        )
    return False


def combined_access_token() -> str | None:
    """Return APP_ID:JWT form for websocket / helpers."""
    load_dotenv()
    app_id = os.environ.get("FYERS_APP_ID", "").strip()
    token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if not app_id or not token:
        return None
    if ":" in token:
        return token
    return f"{app_id}:{token}"


def rest_access_token() -> str | None:
    """JWT-only token for FyersModel REST client."""
    combined = combined_access_token()
    if not combined:
        return None
    return combined.split(":", 1)[-1]


def ensure_valid_access_token(
    *,
    skew_seconds: int = 300,
    force: bool = False,
    remint_fn: Callable[[], dict[str, Any]] | None = None,
) -> str:
    """
    Ensure `.env` access token is usable; remint via TOTP/refresh if expired
    or within skew_seconds of expiry. Returns combined APP_ID:JWT.
    """
    load_dotenv()
    remint = remint_fn or (lambda: refresh_access_token(allow_totp_fallback=True))

    def _current() -> str | None:
        return combined_access_token()

    cur = _current()
    left = token_seconds_left(cur) if cur else None
    need = force or not cur or left is None or left <= float(skew_seconds)
    if not need and cur:
        return cur

    with _remint_lock:
        # Re-check after lock (another thread may have reminted)
        cur = _current()
        left = token_seconds_left(cur) if cur else None
        need = force or not cur or left is None or left <= float(skew_seconds)
        if not need and cur:
            return cur
        logger.warning(
            "Fyers token remint (force=%s left=%s skew=%s)",
            force,
            None if left is None else round(left),
            skew_seconds,
        )
        remint()
        load_dotenv()
        out = _current()
        if not out:
            raise RuntimeError("Remint succeeded but FYERS_ACCESS_TOKEN still missing")
        return out


def fyers_rest_call(method: str, *args: Any, skew_seconds: int = 300, **kwargs: Any) -> Any:
    """
    Call FyersModel method with proactive token ensure + one auth-error remint retry.
    Example: fyers_rest_call("quotes", {"symbols": "NSE:RELIANCE-EQ"})
    """
    from fyers_apiv3 import fyersModel

    ensure_valid_access_token(skew_seconds=skew_seconds)
    app_id = require_env("FYERS_APP_ID")
    jwt = rest_access_token()
    if not jwt:
        raise RuntimeError("No Fyers access token after ensure")
    client = fyersModel.FyersModel(client_id=app_id, token=jwt, is_async=False, log_path="")
    fn = getattr(client, method)
    resp = fn(*args, **kwargs)
    if not is_fyers_auth_error(resp):
        return resp
    logger.warning("Fyers %s auth error — reminting and retrying once", method)
    ensure_valid_access_token(force=True, skew_seconds=skew_seconds)
    jwt = rest_access_token()
    client = fyersModel.FyersModel(client_id=app_id, token=jwt, is_async=False, log_path="")
    return getattr(client, method)(*args, **kwargs)


def auth_url() -> str:
    """Login URL for one-time (or ~15-day) browser authorization."""
    _, _, session = _session()
    return session.generate_authcode()


def _persist_tokens(app_id: str, access_token: str, refresh_token: str | None) -> dict[str, str]:
    # Store combined form expected by Fyers websocket helpers
    combined = access_token if ":" in access_token else f"{app_id}:{access_token}"
    updates = {"FYERS_ACCESS_TOKEN": combined}
    if refresh_token:
        updates["FYERS_REFRESH_TOKEN"] = refresh_token
    upsert_env(updates)
    return updates


def _fyers_client_id() -> str:
    """Fyers user id (FY_ID), not the API app id."""
    load_dotenv()
    for key in ("FYERS_CLIENT_ID", "FYERS_FY_ID", "FYERS_USER_ID"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    raise RuntimeError(
        f"Missing FYERS_CLIENT_ID in {ENV_FILE} "
        "(your Fyers login id, e.g. from get_profile fy_id)."
    )


def exchange_auth_code(auth_code_or_url: str) -> dict[str, Any]:
    """
    Exchange OAuth auth_code for access_token (+ refresh_token).
    Writes tokens into `.env`.
    """
    app_id, _, session = _session()
    code = extract_auth_code(auth_code_or_url)
    if not code:
        raise RuntimeError("Empty auth_code")
    session.set_token(code)
    resp = session.generate_token()
    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected token response: {resp!r}")
    if resp.get("s") == "error" or resp.get("code", 0) not in (0, 200, None):
        if resp.get("s") == "error" or "access_token" not in resp:
            raise RuntimeError(f"Fyers token exchange failed: {resp}")

    access = resp.get("access_token")
    refresh = resp.get("refresh_token")
    if not access:
        raise RuntimeError(f"No access_token in response: {resp}")

    _persist_tokens(app_id, str(access), str(refresh) if refresh else None)
    return {
        "ok": True,
        "has_access_token": True,
        "has_refresh_token": bool(refresh),
        "access_token_len": len(str(access)),
        "env_file": str(ENV_FILE),
        "method": "auth_code",
    }


def login_with_totp(*, pin: str | None = None) -> dict[str, Any]:
    """
    Fully automate daily access token via TOTP + PIN (SEBI-safe path).
    Requires FYERS_TOTP_SECRET, FYERS_PIN, FYERS_CLIENT_ID, app id + secret.
    """
    import pyotp

    load_dotenv()
    app_id, secret, redirect = _creds()
    fy_id = _fyers_client_id()
    totp_secret = os.environ.get("FYERS_TOTP_SECRET", "").strip() or os.environ.get(
        "FYERS_TOTP_KEY", ""
    ).strip()
    if not totp_secret:
        raise RuntimeError(f"Missing FYERS_TOTP_SECRET in {ENV_FILE}")
    pin_val = (pin or os.environ.get("FYERS_PIN", "")).strip()
    if not pin_val:
        raise RuntimeError(f"Missing FYERS_PIN in {ENV_FILE}")

    # Avoid OTP expiry near the 30s boundary
    remaining = 30 - (time.time() % 30)
    if remaining < 5:
        time.sleep(remaining + 0.2)

    r1 = requests.post(
        f"{VAGATOR}/send_login_otp_v2",
        json={"fy_id": _b64(fy_id), "app_id": "2"},
        timeout=30,
    )
    j1 = r1.json()
    if "request_key" not in j1:
        raise RuntimeError(f"TOTP login step1 failed: {j1}")

    otp = pyotp.TOTP(totp_secret).now()
    r2 = requests.post(
        f"{VAGATOR}/verify_otp",
        json={"request_key": j1["request_key"], "otp": otp},
        timeout=30,
    )
    j2 = r2.json()
    if "request_key" not in j2:
        raise RuntimeError(f"TOTP login step2 failed: {j2}")

    r3 = requests.post(
        f"{VAGATOR}/verify_pin_v2",
        json={
            "request_key": j2["request_key"],
            "identity_type": "pin",
            "identifier": _b64(pin_val),
        },
        timeout=30,
    )
    j3 = r3.json()
    bearer = (j3.get("data") or {}).get("access_token")
    if not bearer:
        raise RuntimeError(f"TOTP login step3 (PIN) failed: {j3}")

    if "-" not in app_id:
        raise RuntimeError(f"Unexpected FYERS_APP_ID format: {app_id!r}")
    app_id_prefix, _, app_type = app_id.rpartition("-")
    r4 = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        json={
            "fyers_id": fy_id,
            "app_id": app_id_prefix,
            "redirect_uri": redirect,
            "appType": app_type,
            "code_challenge": "",
            "state": "nse-trader",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        },
        timeout=30,
        allow_redirects=False,
    )
    try:
        j4 = r4.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"TOTP login step4 bad response: {r4.status_code} {r4.text[:200]}"
        ) from exc

    url = j4.get("Url") or j4.get("url") or ""
    if not url:
        raise RuntimeError(f"TOTP login step4 failed: {j4}")
    auth_code = parse_qs(urlparse(url).query).get("auth_code", [None])[0]
    if not auth_code:
        raise RuntimeError(f"TOTP login step4: no auth_code in {url!r}")

    result = exchange_auth_code(auth_code)
    result["method"] = "totp"
    return result


def refresh_access_token(*, pin: str | None = None, allow_totp_fallback: bool = True) -> dict[str, Any]:
    """
    Prefer refresh_token + PIN; if SEBI-disabled, fall back to TOTP login.
    """
    load_dotenv()
    migrate_secret_misplaced_as_access_token()
    app_id, secret, _ = _creds()
    refresh = os.environ.get("FYERS_REFRESH_TOKEN", "").strip()
    pin_val = (pin or os.environ.get("FYERS_PIN", "")).strip()

    if refresh and pin_val:
        payload = {
            "grant_type": "refresh_token",
            "appIdHash": _app_id_hash(app_id, secret),
            "refresh_token": refresh,
            "pin": pin_val,
        }
        r = requests.post(REFRESH_URL, json=payload, timeout=30)
        resp = r.json()
        if resp.get("s") != "error" and "access_token" in resp:
            access = str(resp["access_token"])
            new_refresh = resp.get("refresh_token")
            _persist_tokens(app_id, access, str(new_refresh) if new_refresh else refresh)
            return {
                "ok": True,
                "access_token_len": len(access),
                "refresh_rotated": bool(new_refresh),
                "env_file": str(ENV_FILE),
                "method": "refresh_token",
            }
        msg = str(resp.get("message", ""))
        disabled = resp.get("code") == -16 or "disabled" in msg.lower()
        if not (allow_totp_fallback and disabled):
            raise RuntimeError(f"Fyers refresh failed: {resp}")
        logger.warning("refresh API disabled (%s) — falling back to TOTP login", resp.get("code"))
    elif not allow_totp_fallback:
        raise RuntimeError(
            f"Missing FYERS_REFRESH_TOKEN / FYERS_PIN in {ENV_FILE}. "
            "Run: python main.py fyers-login  (once)"
        )

    return login_with_totp(pin=pin_val or None)


def token_status() -> dict[str, Any]:
    load_dotenv()
    migrated = migrate_secret_misplaced_as_access_token()
    access = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    left = token_seconds_left(access) if access else None
    return {
        "env_file": str(ENV_FILE),
        "has_app_id": bool(os.environ.get("FYERS_APP_ID", "").strip()),
        "has_secret": bool(os.environ.get("FYERS_SECRET_KEY", "").strip()),
        "has_access_token": looks_like_access_token(access),
        "has_refresh_token": bool(os.environ.get("FYERS_REFRESH_TOKEN", "").strip()),
        "has_pin": bool(os.environ.get("FYERS_PIN", "").strip()),
        "has_totp_secret": bool(
            os.environ.get("FYERS_TOTP_SECRET", "").strip()
            or os.environ.get("FYERS_TOTP_KEY", "").strip()
        ),
        "has_client_id": bool(
            os.environ.get("FYERS_CLIENT_ID", "").strip()
            or os.environ.get("FYERS_FY_ID", "").strip()
            or os.environ.get("FYERS_USER_ID", "").strip()
        ),
        "token_seconds_left": None if left is None else int(left),
        "migrated_secret_from_access_slot": migrated,
        "access_token_len": len(access) if access else 0,
    }


def run_login_interactive(auth_code: str | None = None) -> dict[str, Any]:
    """Print auth URL; exchange code from arg or stdin."""
    url = auth_url()
    print("1) Open this URL in a browser, log in, approve the app:")
    print(url)
    print()
    print("2) After redirect, copy the auth_code (or full redirect URL).")
    if not auth_code:
        auth_code = input("Paste auth_code or redirect URL: ").strip()
    result = exchange_auth_code(auth_code)
    print(
        f"OK — wrote access token"
        f"{' + refresh token' if result['has_refresh_token'] else ''} → {result['env_file']}"
    )
    print("For daily auto-login (refresh API is SEBI-disabled), set:")
    print("  FYERS_CLIENT_ID, FYERS_PIN, FYERS_TOTP_SECRET")
    print("Then: python main.py fyers-refresh   # falls back to TOTP")
    return result
