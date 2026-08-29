"""Dashboard payload / HTML smoke."""

from __future__ import annotations


def test_write_dashboard(tmp_path, monkeypatch):
    from ops import dashboard as dash

    out = tmp_path / "dashboard.html"
    monkeypatch.setattr(dash, "DASHBOARD_HTML", out)
    # Use real artifacts if present; still must write HTML
    path = dash.write_dashboard(out)
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "NSE Trader" in html
    assert "DATA =" in html
    assert "eod_markers" in html or "EOD" in html
    assert "metrics" in html.lower()
    assert "offline_metrics" in html or "Metrics table" in html


def test_build_payload_has_keys():
    from ops.dashboard import build_dashboard_payload

    p = build_dashboard_payload()
    for k in (
        "generated_at",
        "glance",
        "eod_markers",
        "series",
        "offline_summary",
        "offline_metrics",
        "forward_metrics",
    ):
        assert k in p
    assert "offline" in p["series"] and "forward" in p["series"]
    assert isinstance(p["offline_metrics"], dict)
    assert isinstance(p["forward_metrics"], dict)
