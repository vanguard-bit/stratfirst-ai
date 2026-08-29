from __future__ import annotations

import re
from pathlib import Path

from ops.monitor.models import Finding, HealthReport, Severity

# Lines that indicate hard failures in service logs.
_ERROR_PATTERNS = [
    re.compile(r"\bERROR\b", re.I),
    re.compile(r"\bCRITICAL\b", re.I),
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bException\b"),
    re.compile(r"\bFATAL\b", re.I),
    re.compile(r"NotImplementedError"),
]

# Benign lines that match error patterns but are not failures.
_IGNORE_PATTERNS = [
    re.compile(r"no errors?", re.I),
    re.compile(r"0 errors?", re.I),
    # Fyers WS teardown / DNS blips — common at session end, not app faults
    re.compile(r"fyers ws error", re.I),
    re.compile(r"close_connection still blocked", re.I),
    re.compile(r"Temporary failure in name resolution", re.I),
    re.compile(r"Connection timed out", re.I),
]


def scan_logs(logs_dir: Path, max_errors: int = 0) -> HealthReport:
    report = HealthReport()

    if not logs_dir.exists():
        report.add(
            Finding(
                check="logs_present",
                severity=Severity.WARN,
                message=f"Log directory missing: {logs_dir}",
            )
        )
        return report

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        report.add(
            Finding(
                check="logs_present",
                severity=Severity.WARN,
                message=f"No *.log files in {logs_dir} (service may not have run yet)",
            )
        )
        return report

    hits: list[dict] = []
    for path in log_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report.add(
                Finding(
                    check="log_readable",
                    severity=Severity.ERROR,
                    message=f"Cannot read log {path.name}: {exc}",
                )
            )
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or any(p.search(stripped) for p in _IGNORE_PATTERNS):
                continue
            if any(p.search(stripped) for p in _ERROR_PATTERNS):
                hits.append({"file": path.name, "line": lineno, "text": stripped[:200]})

    if not hits:
        report.add(
            Finding(
                check="log_errors",
                severity=Severity.OK,
                message=f"Scanned {len(log_files)} log file(s) — no error patterns",
                detail={"files": [p.name for p in log_files]},
            )
        )
        return report

    severity = Severity.ERROR if len(hits) > max_errors else Severity.WARN
    report.add(
        Finding(
            check="log_errors",
            severity=severity,
            message=f"Found {len(hits)} error-like line(s) in logs (limit {max_errors})",
            detail={"hits": hits[:20], "truncated": len(hits) > 20},
        )
    )
    return report
