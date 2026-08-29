"""Upsert helpers for project `.env` (never log secret values)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from nse_trader.env import ENV_FILE, load_dotenv


def upsert_env(updates: dict[str, str], path: Path | None = None) -> Path:
    """
    Create or update keys in `.env`. Preserves comments and unrelated keys.
    Also updates os.environ for the current process.
    """
    env_path = path or ENV_FILE
    load_dotenv(env_path)
    existing: list[str] = []
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, _, _ = line.partition("=")
        key = key.strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
            os.environ[key] = updates[key]
        else:
            out.append(line)

    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
            os.environ[key] = value

    text = "\n".join(out).rstrip() + "\n"
    env_path.write_text(text, encoding="utf-8")
    return env_path


def looks_like_access_token(value: str) -> bool:
    v = value.strip()
    if ":" in v:
        v = v.split(":", 1)[1]
    return v.startswith("eyJ") and len(v) > 40


def migrate_secret_misplaced_as_access_token() -> bool:
    """
    If FYERS_ACCESS_TOKEN holds the secret (common mistake), move it to FYERS_SECRET_KEY.
    Returns True if migration happened.
    """
    load_dotenv()
    access = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    secret = os.environ.get("FYERS_SECRET_KEY", "").strip()
    if not access or looks_like_access_token(access):
        return False
    if secret:
        # access is wrong; clear it so we don't keep using secret as token
        upsert_env({"FYERS_ACCESS_TOKEN": ""})
        return True
    upsert_env({"FYERS_SECRET_KEY": access, "FYERS_ACCESS_TOKEN": ""})
    return True


_AUTH_CODE_RE = re.compile(r"(?:auth_code|code)=([^&]+)")


def extract_auth_code(text: str) -> str:
    """Accept raw auth_code or a full redirect URL containing it."""
    text = text.strip().strip('"').strip("'")
    m = _AUTH_CODE_RE.search(text)
    if m:
        from urllib.parse import unquote

        return unquote(m.group(1))
    return text
