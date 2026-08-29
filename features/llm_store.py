"""Load latest Gemini LLM feature parquet for live meta / strategy state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nse_trader.config import PortfolioConfig


@dataclass
class LLMMarketSummary:
    mean_sentiment: float = 0.0
    high_materiality: int = 0
    n_symbols: int = 0
    as_of: str = ""
    path: str = ""


def features_dir(store_path: Path | None = None) -> Path:
    root = Path(store_path) if store_path is not None else PortfolioConfig.load().store_path
    return root / "features"


def latest_llm_parquet(store_path: Path | None = None) -> Path | None:
    d = features_dir(store_path)
    if not d.is_dir():
        return None
    files = sorted(d.glob("llm_*.parquet"))
    return files[-1] if files else None


def load_llm_map(store_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Symbol → {sentiment, materiality, events, as_of}.
    Missing / unreadable store → empty dict (callers treat as neutral).
    """
    path = latest_llm_parquet(store_path)
    if path is None:
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return {}
    if df.empty or "symbol" not in df.columns:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper().strip()
        events = row.get("events", [])
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except json.JSONDecodeError:
                events = []
        if not isinstance(events, list):
            events = []
        out[sym] = {
            "sentiment": float(row.get("sentiment", 0.0) or 0.0),
            "materiality": str(row.get("materiality", "low") or "low"),
            "events": events,
            "as_of": str(row.get("as_of", "") or ""),
            "source": str(path),
        }
    return out


def summarize_llm(llm_map: dict[str, dict[str, Any]]) -> LLMMarketSummary:
    if not llm_map:
        return LLMMarketSummary()
    sentiments = [float(v.get("sentiment", 0.0) or 0.0) for v in llm_map.values()]
    high = sum(1 for v in llm_map.values() if str(v.get("materiality", "")).lower() == "high")
    as_ofs = [str(v.get("as_of", "")) for v in llm_map.values() if v.get("as_of")]
    path = next((str(v.get("source", "")) for v in llm_map.values() if v.get("source")), "")
    return LLMMarketSummary(
        mean_sentiment=float(sum(sentiments) / len(sentiments)),
        high_materiality=high,
        n_symbols=len(llm_map),
        as_of=max(as_ofs) if as_ofs else "",
        path=path,
    )


def inject_symbol_llm(state: dict[str, Any], symbol: str, llm_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feat = llm_map.get(str(symbol).upper().strip(), {})
    state["llm_sentiment"] = float(feat.get("sentiment", 0.0) or 0.0)
    state["llm_materiality"] = str(feat.get("materiality", "low") or "low")
    state["llm_events"] = list(feat.get("events", []) or [])
    return state
