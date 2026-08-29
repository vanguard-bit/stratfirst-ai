"""Offline demo command integration test."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.runtime


def test_demo_runs_offline(tmp_path: Path):
    from experiments.demo import run_demo

    result = run_demo(out_dir=tmp_path / "demo")
    assert result["meta_allocator_mode"] == "rules"
    assert result["n_implemented"] == 21
    assert result["n_enabled"] == 13
    assert (tmp_path / "demo" / "dashboard.html").is_file()
    assert (tmp_path / "demo" / "demo_report.json").is_file()
    assert any(c["case"] == "circuit_reject" and not c["allowed"] for c in result["execution_cases"])
    assert any(c["case"] == "fill" and c["allowed"] for c in result["execution_cases"])
    assert "weights_with_llm" in result and "weights_no_llm" in result
