from __future__ import annotations

import json
from pathlib import Path

from ops.monitor.allocator_audit import audit_allocations
from ops.monitor.config import MonitorConfig
from ops.monitor.log_scan import scan_logs
from ops.monitor.models import HealthReport, Severity
from ops.monitor.retention import check_retention_reminder
from ops.monitor.strategy_integrity import check_strategy_config, check_trade_logs


def run_health_checks(cfg: MonitorConfig | None = None) -> HealthReport:
    cfg = cfg or MonitorConfig.load()
    report = HealthReport()
    report.merge(check_strategy_config())
    report.merge(scan_logs(cfg.logs_dir, max_errors=cfg.thresholds.max_log_errors))
    report.merge(audit_allocations(cfg.allocation_history, cfg.thresholds))
    report.merge(check_trade_logs(cfg.trade_glob))
    p = cfg.persistence
    report.merge(check_retention_reminder(p.store_dir, p.logs_dir, p.state_dir))
    return report


def format_report(report: HealthReport) -> str:
    lines = [
        f"Health: {'PASS' if report.ok else 'FAIL'} "
        f"({report.error_count} error(s), {report.warn_count} warning(s))",
        "",
    ]
    for f in report.findings:
        icon = {"ok": "✓", "warn": "!", "error": "✗"}[f.severity.value]
        lines.append(f"  [{icon}] {f.check}: {f.message}")
    return "\n".join(lines)


def write_report_json(report: HealthReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": report.ok,
        "error_count": report.error_count,
        "warn_count": report.warn_count,
        "findings": [
            {
                "check": f.check,
                "severity": f.severity.value,
                "message": f.message,
                "detail": f.detail,
            }
            for f in report.findings
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
