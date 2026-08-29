from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from features.llm_schema import LLMFeatureRow, validate_row
from nse_trader.env import ENV_FILE

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _today() -> str:
    return date.today().isoformat()


def _row_to_dict(row: LLMFeatureRow) -> dict[str, Any]:
    validate_row(row)
    return {
        "symbol": row.symbol,
        "as_of": row.as_of,
        "sentiment": row.sentiment,
        "materiality": row.materiality,
        "events": json.dumps(row.events),
    }


def extract_features_offline_sample(
    symbols: list[str],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministic neutral features — no API call."""
    stamp = as_of or _today()
    return [
        _row_to_dict(
            LLMFeatureRow(
                symbol=symbol,
                as_of=stamp,
                sentiment=0.0,
                materiality="low",
                events=[],
            )
        )
        for symbol in symbols
    ]


def _parse_gemini_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_gemini(
    api_key: str,
    model: str,
    symbol: str,
    headline: str,
    as_of: str,
    *,
    market_context: str = "",
) -> dict[str, Any]:
    ctx = ""
    if market_context.strip():
        ctx = f"Market context:\n{market_context.strip()}\n\n"
    prompt = (
        f"{ctx}Symbol: {symbol}\nDate: {as_of}\nHeadlines:\n{headline or 'No material news.'}\n\n"
        "Return JSON only with keys: sentiment (float -1 to 1), "
        "materiality (low|medium|high), events (list of short strings)."
    )
    url = GEMINI_URL.format(model=model)
    resp = requests.post(
        url,
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    parsed = _parse_gemini_json(text)
    row = LLMFeatureRow(
        symbol=symbol,
        as_of=as_of,
        sentiment=float(parsed.get("sentiment", 0.0)),
        materiality=str(parsed.get("materiality", "low")),
        events=list(parsed.get("events", [])),
    )
    return _row_to_dict(row)


def extract_features_live(
    symbols: list[str],
    headlines: dict[str, str],
    *,
    as_of: str | None = None,
    market_context: str = "",
) -> list[dict[str, Any]]:
    """Batch compress headlines → structured JSON features via Gemini.

    Per-symbol failures fall back to neutral offline rows (never invent events).
    """
    from nse_trader.env import ENV_FILE

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing GEMINI_API_KEY. Set it in {ENV_FILE} (see .env.example)."
        )
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    stamp = as_of or _today()
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        headline = headlines.get(symbol, "")
        try:
            rows.append(
                _call_gemini(
                    api_key,
                    model,
                    symbol,
                    headline,
                    stamp,
                    market_context=market_context,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Honest placeholder — no fabricated sentiment from a failed call
            rows.append(
                _row_to_dict(
                    LLMFeatureRow(
                        symbol=symbol,
                        as_of=stamp,
                        sentiment=0.0,
                        materiality="low",
                        events=[f"extract_error:{type(exc).__name__}"],
                    )
                )
            )
    return rows


def write_features_parquet(rows: list[dict[str, Any]], path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out
