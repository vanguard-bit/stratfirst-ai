"""Phase 6b contract — LLM feature compression."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.phase6b


REQUIRED_LLM_FIELDS = {"symbol", "as_of", "sentiment", "materiality", "events"}


def test_llm_output_schema():
    from features.llm_schema import LLMFeatureRow, validate_row

    row = LLMFeatureRow(
        symbol="RELIANCE",
        as_of="2026-08-10",
        sentiment=0.2,
        materiality="low",
        events=[],
    )
    validate_row(row)


def test_llm_extract_offline_sample():
    from features.llm_gemini import extract_features_offline_sample

    rows = extract_features_offline_sample(["RELIANCE", "TCS"])
    assert len(rows) == 2
    for row in rows:
        assert REQUIRED_LLM_FIELDS.issubset(set(row.keys()))


def test_llm_extract_live_requires_api_key(monkeypatch):
    from features.llm_gemini import extract_features_live

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        extract_features_live(["RELIANCE"], headlines={"RELIANCE": "No news"})


def test_llm_writes_parquet(tmp_path):
    from features.llm_gemini import write_features_parquet

    rows = [
        {
            "symbol": "RELIANCE",
            "as_of": "2026-08-10",
            "sentiment": 0.1,
            "materiality": "low",
            "events": json.dumps([]),
        }
    ]
    path = write_features_parquet(rows, tmp_path / "llm_daily.parquet")
    assert path.exists()
