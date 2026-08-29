"""EOD bake-off isolation: failures must not kill EOD summary."""

from __future__ import annotations

import json
from pathlib import Path


def test_eod_survives_bakeoff_failure(tmp_path, monkeypatch):
    from experiments import eod as eod_mod

    day = "2026-08-11"
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

    (_Cfg.store_path / "experiments" / "eod" / "paper").mkdir(parents=True)
    day_file = _Cfg.store_path / "experiments" / "eod" / "paper" / f"day_{day}.json"
    day_file.write_text(json.dumps({"n_trades": 0, "run_id": f"paper-{day}"}))

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

    import meta.shadow as shadow_mod

    monkeypatch.setattr(
        shadow_mod,
        "run_meta_shadow",
        lambda **kw: {"mode": "lightgbm_shadow", "top5": ["A1"], "top5_reasons": ["x"]},
    )
    monkeypatch.setattr(
        shadow_mod,
        "feature_rows_for_shadow",
        lambda *a, **k: __import__("pandas").DataFrame({"strategy_id": ["A1"], "ret_1d": [0.0]}),
    )

    def _boom(**kwargs):
        raise RuntimeError("measure exploded")

    monkeypatch.setattr(
        "experiments.measure_ledger.aggregate_strat_day",
        _boom,
    )

    summary = eod_mod.run_eod(date=day)
    assert summary["health_ok"] is True
    assert summary.get("meta_bakeoff", {}).get("error")
    assert Path(summary["path"]).exists()


def test_eod_survives_metrics_failure(tmp_path, monkeypatch):
    """Measure/glance succeed; metrics boom → warning only, EOD still healthy."""
    from experiments import eod as eod_mod

    day = "2026-08-11"
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

    (_Cfg.store_path / "experiments" / "eod" / "paper").mkdir(parents=True)
    day_file = _Cfg.store_path / "experiments" / "eod" / "paper" / f"day_{day}.json"
    day_file.write_text(json.dumps({"n_trades": 0, "run_id": f"paper-{day}"}))

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

    import meta.shadow as shadow_mod

    monkeypatch.setattr(
        shadow_mod,
        "run_meta_shadow",
        lambda **kw: {"mode": "lightgbm_shadow", "top5": ["A1"], "top5_reasons": ["x"]},
    )
    monkeypatch.setattr(
        shadow_mod,
        "feature_rows_for_shadow",
        lambda *a, **k: __import__("pandas").DataFrame({"strategy_id": ["A1"], "ret_1d": [0.0]}),
    )
    monkeypatch.setattr(
        "experiments.measure_ledger.aggregate_strat_day",
        lambda **kw: __import__("pandas").DataFrame(),
    )

    def _metrics_boom(**kwargs):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(
        "experiments.metrics_table.build_forward_metrics",
        _metrics_boom,
    )
    monkeypatch.setattr(
        "ops.dashboard.write_dashboard",
        lambda *a, **k: None,
    )

    # Point bakeoff paths at tmp so append does not touch real data
    import experiments.meta_bakeoff as mb

    monkeypatch.setattr(mb, "BAKEOFF_DIR", tmp_path / "bakeoff")
    monkeypatch.setattr(mb, "FORWARD_DAILY", tmp_path / "bakeoff" / "forward_daily.parquet")
    monkeypatch.setattr(mb, "STRAT_DAILY", tmp_path / "bakeoff" / "strat.parquet")
    monkeypatch.setattr(mb, "GLANCE_PATH", tmp_path / "state" / "meta_bakeoff_glance.json")
    (tmp_path / "bakeoff").mkdir(parents=True)

    summary = eod_mod.run_eod(date=day)
    assert summary["health_ok"] is True
    assert Path(summary["path"]).exists()
    warns = (summary.get("meta_bakeoff") or {}).get("warnings") or []
    assert any("forward metrics failed" in str(w) for w in warns)
