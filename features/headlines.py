"""Best-effort headline fetch for LLM feature extraction (fail-soft)."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)


def fetch_google_news_headlines(symbol: str, *, limit: int = 3, timeout: float = 8.0) -> str:
    """
    Pull a few Google News RSS titles for `SYMBOL NSE stock`.
    Returns empty string on any failure (offline / blocked / parse error).
    """
    q = quote_plus(f"{symbol} NSE stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "nse-trader/1.0 (+local research)"},
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.debug("headline fetch failed for %s: %s", symbol, exc)
        return ""

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return ""

    titles: list[str] = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is None or not (title_el.text or "").strip():
            continue
        title = re.sub(r"\s+", " ", title_el.text.strip())
        if title.lower().startswith("google news"):
            continue
        titles.append(title)
        if len(titles) >= limit:
            break
    return "\n".join(titles)


def fetch_headlines_map(symbols: list[str], *, limit: int = 3) -> dict[str, str]:
    out: dict[str, str] = {}
    for sym in symbols:
        text = fetch_google_news_headlines(sym, limit=limit)
        out[sym] = text if text else "No material news."
    return out
