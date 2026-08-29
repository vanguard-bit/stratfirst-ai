from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import requests

from nse_trader.config import ROOT

# Prevent indefinite DNS/TCP hangs
socket.setdefaulttimeout(8)

DEFAULT_TIMEOUT = (4, 8)  # connect, read seconds

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_url(url: str) -> requests.Response:
    resp = requests.get(url, headers=BROWSER_HEADERS, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp
