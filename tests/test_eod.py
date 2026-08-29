"""EOD CLI + market-hours helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ops.market_hours import in_ingest_window


def test_in_ingest_window_weekday_noon():
    ts = datetime(2026, 8, 10, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Mon
    assert in_ingest_window(ts) is True


def test_in_ingest_window_before_open():
    ts = datetime(2026, 8, 10, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert in_ingest_window(ts) is False


def test_eod_writes_summary(tmp_path, monkeypatch):
    """Must not touch live logs, measure books, or glance."""
    import json
    from pathlib import Path

    from experiments import eod as eod_mod

    day = "2026-08-10"
    logs = tmp_path / "logs"
    logs.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "portfolio.json").write_text("{}")

    monkeypatch.setattr(
        eod_mod,
        "load_yaml",
        lambda name: {
            "persistence": {
                "logs_dir": str(logs),
                "state_dir": str(state),
            }
        }
        if name == "ops.yaml"
        else {},
    )

    class _Cfg:
        store_path = tmp_path / "store"

    paper_dir = _Cfg.store_path / "experiments" / "eod" / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / f"day_{day}.json").write_text(
        json.dumps({"n_trades": 0, "run_id": f"paper-{day}"})
    )

    monkeypatch.setattr(eod_mod.PortfolioConfig, "load", staticmethod(lambda: _Cfg()))
    monkeypatch.setattr(eod_mod, "reconcile_state", lambda *a, **k: None)
    monkeypatch.setattr(
        eod_mod,
        "run_health_checks",
        lambda: type(
            "HR",
            (),
            {"ok": True, "warn_count": 0, "error_count": 0, "findings": []},
        )(),
    )
    monkeypatch.setattr(eod_mod, "write_report_json", lambda *a, **k: None)
    monkeypatch.setattr(
        "experiments.measure_ledger.aggregate_strat_day",
        lambda **kw: __import__("pandas").DataFrame(),
    )
    monkeypatch.setattr(
        "experiments.meta_bakeoff.refresh_forward_and_glance",
        lambda **kw: {"as_of": day, "ml_top5": [], "warnings": []},
    )
    monkeypatch.setattr(
        "meta.shadow.run_meta_shadow",
        lambda **kw: {"mode": "lightgbm_shadow", "top5": []},
    )

    summary = eod_mod.run_eod(date=day)
    assert summary["date"] == day
    assert "n_trades" in summary
    assert summary["path"].endswith(f"eod_{day}.json")
    assert Path(summary["path"]).exists()
    assert summary["health_ok"] is True
    live_glance = Path("data/state/meta_bakeoff_glance.json")
    if live_glance.exists():
        assert json.loads(live_glance.read_text(encoding="utf-8")).get("as_of") != day
