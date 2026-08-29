"""Public glance export for GitHub Pages."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.public_glance import build_public_glance, export_public_glance

pytestmark = pytest.mark.runtime


def test_public_glance_has_no_home_paths():
    payload = build_public_glance(include_local=True)
    blob = str(payload)
    assert "/home/" not in blob
    assert "100.108." not in blob
    assert payload["paper_only"] is True
    assert payload["lightgbm"] == "shadow_only"
    assert payload["offline"]["mean_auc"] == 0.629


def test_export_public_glance_writes_site(tmp_path: Path):
    payload = export_public_glance(include_local=False, site_dir=tmp_path / "site")
    assert (tmp_path / "site" / "glance.json").is_file()
    assert (tmp_path / "site" / "index.html").is_file()
    assert (tmp_path / "site" / ".nojekyll").is_file()
    assert payload["_paths"]["glance_json"].endswith("glance.json")
