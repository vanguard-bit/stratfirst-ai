"""Diagnose harness for coding agents."""

from __future__ import annotations

from ops.monitor.diagnose import audit_artifacts, audit_systemd, format_diagnose, run_diagnose
from ops.monitor.models import Severity


def test_audit_systemd_missing_unit():
    info = {
        "available": True,
        "units": {
            "nse-trader-ingest.timer": {"ok": False, "error": "not found"},
            "nse-trader-paper.timer": {
                "ok": True,
                "load_state": "loaded",
                "active_state": "active",
                "unit_file_state": "enabled",
            },
        },
        "timers_table": "",
        "errors": [],
    }
    report = audit_systemd(info)
    assert any(f.check == "systemd_units_loaded" and f.severity == Severity.ERROR for f in report.findings)


def test_audit_artifacts_missing_eod():
    report = audit_artifacts({"day": "2099-01-01", "eod": None, "log_files": []})
    assert any(f.check == "eod_artifact" and f.severity == Severity.WARN for f in report.findings)


def test_count_journal_timeouts():
    from ops.monitor.diagnose import count_journal_timeouts

    text = (
        "nse-trader-ingest.service: Failed with result 'timeout'.\n"
        "Finished nse-trader-ingest.service\n"
        "nse-trader-ingest.service: Failed with result 'timeout'.\n"
    )
    assert count_journal_timeouts(text) == 2
    assert count_journal_timeouts("Finished ok") == 0


def test_audit_systemd_ingest_timeouts_today():
    info = {
        "available": True,
        "units": {
            "nse-trader-ingest.timer": {
                "ok": True,
                "load_state": "loaded",
                "active_state": "active",
                "unit_file_state": "enabled",
            },
            "nse-trader-ingest.service": {
                "ok": True,
                "load_state": "loaded",
                "active_state": "inactive",
                "result": "success",
            },
        },
        "ingest_timeouts": {"day": "2026-08-18", "count": 6, "available": True},
        "timers_table": "",
        "errors": [],
    }
    report = audit_systemd(info)
    assert any(
        f.check == "systemd_ingest_timeouts" and f.severity == Severity.ERROR for f in report.findings
    )


def test_run_diagnose_smoke():
    payload = run_diagnose(day="2026-08-10")
    assert "ok" in payload
    assert "findings" in payload
    assert "systemd" in payload
    assert "agent_next_steps" in payload
    text = format_diagnose(payload)
    assert "Diagnose:" in text
