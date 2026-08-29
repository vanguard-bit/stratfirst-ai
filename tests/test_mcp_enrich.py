"""MCP enrichment compressors + budgeted fetch_market_context."""

from __future__ import annotations

from pathlib import Path

from features.mcp_enrich import (
    bust_meta_weights_cache,
    compress_market_news,
    compress_morning_brief,
    fetch_market_context,
)


def test_compress_morning_brief_plain():
    payload = {
        "morning_text": "Good morning. VIX low. Direction bullish.",
        "pre_market": {"india_vix": {"current_vix": 11.0}},
    }
    text = compress_morning_brief(payload)
    assert "Good morning" in text
    assert len(text) <= 1200


def test_compress_market_news_list():
    payload = {
        "articles": [
            {
                "title": "RBI holds rates",
                "sentiment": "neutral",
                "symbols": ["HDFCBANK", "ICICIBANK"],
            },
            {"title": "US futures up", "sentiment": "positive", "symbols": []},
        ]
    }
    text = compress_market_news(payload, limit=10)
    assert "RBI holds rates" in text
    assert "HDFCBANK" in text
    assert "[neutral]" in text


def test_fetch_market_context_uses_injectables_and_budget(tmp_path: Path):
    state = tmp_path / "budget.json"
    log = tmp_path / "budget.jsonl"
    cfg = {
        "enabled": True,
        "max_calls_per_day": 2,
        "max_calls_per_month": 40,
        "finstack": True,
        "tapetide": True,
    }
    ctx = fetch_market_context(
        cfg=cfg,
        finstack_fn=lambda: "brief-ok",
        tapetide_fn=lambda: "news-ok",
        state_path=state,
        log_path=log,
    )
    assert "brief-ok" in ctx
    assert "news-ok" in ctx
    # second full fetch should be blocked (2/2 used)
    ctx2 = fetch_market_context(
        cfg=cfg,
        finstack_fn=lambda: "brief-2",
        tapetide_fn=lambda: "news-2",
        state_path=state,
        log_path=log,
    )
    assert ctx2 == ""


def test_fetch_market_context_fail_soft(tmp_path: Path):
    state = tmp_path / "budget.json"
    log = tmp_path / "budget.jsonl"
    cfg = {
        "enabled": True,
        "max_calls_per_day": 2,
        "max_calls_per_month": 40,
        "finstack": True,
        "tapetide": False,
    }

    def boom():
        raise RuntimeError("down")

    ctx = fetch_market_context(
        cfg=cfg,
        finstack_fn=boom,
        state_path=state,
        log_path=log,
    )
    assert ctx == ""
    # spend still recorded
    assert state.exists()


def test_bust_meta_weights_cache(tmp_path: Path):
    p = tmp_path / "meta_weights_day.json"
    p.write_text("{}", encoding="utf-8")
    assert bust_meta_weights_cache(p) is True
    assert not p.exists()
    assert bust_meta_weights_cache(p) is False
