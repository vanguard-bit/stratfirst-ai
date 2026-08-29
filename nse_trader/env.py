"""Load key=value pairs from project root `.env` file."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load_dotenv(path: Path | None = None) -> None:
    """Parse `.env` into os.environ.

    Does not override non-empty existing vars. Empty/whitespace env values
    are replaced (so a broken systemd EnvironmentFile cannot block secrets).
    """
    env_path = path or ENV_FILE
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        existing = os.environ.get(key)
        if existing is None or not str(existing).strip():
            os.environ[key] = value


# backwards compat alias
load_environment = load_dotenv


def require_env(name: str) -> str:
    load_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Set it in {ENV_FILE} (see .env.example)."
        )
    return value
