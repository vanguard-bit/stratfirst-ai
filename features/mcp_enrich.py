"""Budgeted FinStack / Tapetide MCP enrichment for morning llm-extract."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from features.mcp_budget import BudgetConfig, can_spend, record_spend
from nse_trader.config import load_yaml

logger = logging.getLogger(__name__)

EnrichFn = Callable[[], str]


def load_enrichment_config() -> dict[str, Any]:
    try:
        ops = load_yaml("ops.yaml")
    except Exception:  # noqa: BLE001
        ops = {}
    raw = ops.get("llm_enrichment") or {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_calls_per_day": int(raw.get("max_calls_per_day", 2)),
        "max_calls_per_month": int(raw.get("max_calls_per_month", 40)),
        "finstack": bool(raw.get("finstack", True)),
        "tapetide": bool(raw.get("tapetide", True)),
        "news_limit": int(raw.get("news_limit", 15)),
    }


def compress_morning_brief(payload: Any) -> str:
    """Turn FinStack morning brief JSON into a short prompt blurb."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload.strip()
        return text[:1200] if text else ""
    if not isinstance(payload, dict):
        return str(payload)[:800]

    parts: list[str] = []
    morning = payload.get("morning_text") or (payload.get("delivery_formats") or {}).get("plain_text")
    if morning:
        parts.append(str(morning).strip()[:900])
    else:
        pre = payload.get("pre_market") or {}
        vix = pre.get("india_vix") or {}
        if vix.get("current_vix") is not None:
            parts.append(
                f"India VIX={vix.get('current_vix')} ({vix.get('signal') or ''})".strip()
            )
        direction = pre.get("nifty_direction") or {}
        if direction.get("probability_up") is not None:
            parts.append(
                f"Nifty direction={direction.get('signal')} "
                f"({direction.get('probability_up')}% up)"
            )
        flows = payload.get("institutional_flow") or {}
        for row in flows.get("data") or []:
            parts.append(
                f"{row.get('category')} net={row.get('netValue')} on {row.get('date')}"
            )
        movers = payload.get("market_movers") or {}
        gainers = movers.get("gainers") or []
        losers = movers.get("losers") or []
        if gainers:
            g = ", ".join(f"{x.get('symbol')} {x.get('change_pct')}%" for x in gainers[:3])
            parts.append(f"Gainers: {g}")
        if losers:
            l = ", ".join(f"{x.get('symbol')} {x.get('change_pct')}%" for x in losers[:3])
            parts.append(f"Losers: {l}")
    return " | ".join(p for p in parts if p)[:1200]


def compress_market_news(payload: Any, *, limit: int = 15) -> str:
    """Turn Tapetide market news into short lines with symbols + sentiment."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()[:1200]
    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("news")
            or payload.get("articles")
            or payload.get("items")
            or payload.get("data")
            or []
        )
        if not items and payload.get("title"):
            items = [payload]
    else:
        return str(payload)[:800]

    lines: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            lines.append(str(item)[:160])
            continue
        title = str(item.get("title") or item.get("headline") or "").strip()
        sent = str(item.get("sentiment") or "").strip()
        syms = item.get("symbols") or item.get("related_symbols") or item.get("tickers") or []
        if isinstance(syms, str):
            syms = [syms]
        sym_s = ",".join(str(s) for s in list(syms)[:4])
        bit = title
        if sent:
            bit = f"[{sent}] {bit}"
        if sym_s:
            bit = f"{bit} ({sym_s})"
        if bit:
            lines.append(bit[:200])
    return "\n".join(lines)[:1200]


def _tool_text(result: Any) -> Any:
    """Normalize MCP CallToolResult / dict / str into JSON-able payload."""
    if result is None:
        return None
    if isinstance(result, (dict, list, str)):
        return result
    # MCP SDK CallToolResult
    content = getattr(result, "content", None)
    if content:
        chunks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        joined = "\n".join(chunks).strip()
        if not joined:
            return None
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    return str(result)


async def _mcp_call_tool(
    *,
    command: str,
    args: list[str],
    env: dict[str, str],
    tool: str,
    arguments: dict[str, Any] | None = None,
    timeout_sec: float = 90.0,
) -> Any:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool(tool, arguments or {}),
                timeout=timeout_sec,
            )
            return _tool_text(result)


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    # Prefer the caller's PATH; only seed a minimal fallback when unset.
    home_local = Path.home() / ".local" / "bin"
    cargo_bin = Path.home() / ".cargo" / "bin"
    fallback = os.pathsep.join(
        ["/usr/local/bin", "/usr/bin", "/bin", str(home_local), str(cargo_bin)]
    )
    env.setdefault("PATH", fallback)
    return env


def fetch_finstack_morning_brief() -> str:
    payload = asyncio.run(
        _mcp_call_tool(
            command="uvx",
            args=["--with", "mcp>=1.2,<2", "finstack-mcp"],
            env=_base_env(),
            tool="get_morning_brief",
            arguments={},
        )
    )
    return compress_morning_brief(payload)


def fetch_tapetide_market_news(*, limit: int = 15) -> str:
    token = os.environ.get("TAPETIDE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TAPETIDE_TOKEN in environment")
    env = _base_env()
    env["TAPETIDE_TOKEN"] = token
    payload = asyncio.run(
        _mcp_call_tool(
            command="npx",
            args=["-y", "tapetide-mcp"],
            env=env,
            tool="get_market_news",
            arguments={"limit": limit},
        )
    )
    return compress_market_news(payload, limit=limit)


def fetch_market_context(
    *,
    cfg: dict[str, Any] | None = None,
    finstack_fn: EnrichFn | None = None,
    tapetide_fn: EnrichFn | None = None,
    state_path: Path | None = None,
    log_path: Path | None = None,
) -> str:
    """
    Budgeted market context for Gemini prompts.
    Fail-soft: returns "" when disabled, over budget, or providers error.
    """
    cfg = cfg or load_enrichment_config()
    if not cfg.get("enabled", True):
        return ""

    budget = BudgetConfig(
        max_calls_per_day=int(cfg.get("max_calls_per_day", 2)),
        max_calls_per_month=int(cfg.get("max_calls_per_month", 40)),
    )
    want_f = bool(cfg.get("finstack", True))
    want_t = bool(cfg.get("tapetide", True))
    if not want_f and not want_t:
        return ""
    if not can_spend(1, budget, state_path=state_path):
        logger.warning("MCP daily/monthly budget exhausted — skipping enrichment")
        return ""

    finstack_fn = finstack_fn or fetch_finstack_morning_brief
    tapetide_fn = tapetide_fn or fetch_tapetide_market_news
    chunks: list[str] = []

    if want_f and can_spend(1, budget, state_path=state_path):
        try:
            text = finstack_fn()
            record_spend(
                provider="finstack",
                tool="get_morning_brief",
                cfg=budget,
                state_path=state_path,
                log_path=log_path,
                ok=bool(text),
                detail=(text or "")[:200],
            )
            if text:
                chunks.append(f"FinStack brief: {text}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("finstack enrich failed: %s", exc)
            record_spend(
                provider="finstack",
                tool="get_morning_brief",
                cfg=budget,
                state_path=state_path,
                log_path=log_path,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    if want_t and can_spend(1, budget, state_path=state_path):
        try:
            if tapetide_fn is fetch_tapetide_market_news:
                text = fetch_tapetide_market_news(limit=int(cfg.get("news_limit", 15)))
            else:
                text = tapetide_fn()
            record_spend(
                provider="tapetide",
                tool="get_market_news",
                cfg=budget,
                state_path=state_path,
                log_path=log_path,
                ok=bool(text),
                detail=(text or "")[:200],
            )
            if text:
                chunks.append(f"Tapetide news:\n{text}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("tapetide enrich failed: %s", exc)
            record_spend(
                provider="tapetide",
                tool="get_market_news",
                cfg=budget,
                state_path=state_path,
                log_path=log_path,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    return "\n\n".join(chunks).strip()


def bust_meta_weights_cache(path: Path | None = None) -> bool:
    """Delete daily meta weights cache so next paper tick recomputes dual dump."""
    from meta.regime import WEIGHTS_CACHE

    p = Path(path or WEIGHTS_CACHE)
    if not p.exists():
        return False
    p.unlink()
    return True
