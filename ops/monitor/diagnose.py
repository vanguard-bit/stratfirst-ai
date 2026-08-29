"""Agent-facing diagnose: health + systemd timers/services + recent artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nse_trader.config import ROOT, load_yaml
from ops.monitor.config import MonitorConfig
from ops.monitor.models import Finding, HealthReport, Severity
from ops.monitor.runner import format_report, run_health_checks, write_report_json

IST = ZoneInfo("Asia/Kolkata")

UNITS = (
    "nse-trader-ingest.timer",
    "nse-trader-ingest.service",
    "nse-trader-paper.timer",
    "nse-trader-paper.service",
    "nse-trader-eod.timer",
    "nse-trader-eod.service",
    "nse-trader-fyers-refresh.timer",
    "nse-trader-fyers-refresh.service",
    "nse-trader-llm.timer",
    "nse-trader-llm.service",
    "nse-trader-dashboard.service",
)

_PROP_KEYS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "Result",
    "ExecMainStatus",
    "ExecMainCode",
    "InactiveExitTimestamp",
    "ActiveEnterTimestamp",
    "NextElapseUSecRealtime",
    "LastTriggerUSec",
    "FragmentPath",
)


_TIMEOUT_RESULT = "Failed with result 'timeout'"


def count_journal_timeouts(text: str) -> int:
    """Count systemd oneshot kills: Failed with result 'timeout'."""
    if not text:
        return 0
    return text.count(_TIMEOUT_RESULT)


def collect_ingest_timeouts(*, day: str | None = None) -> dict:
    """Today's ingest timeouts from the user journal (last Result is not enough)."""
    day = day or datetime.now(tz=IST).date().isoformat()
    since = f"{day} 00:00:00"
    cmd = [
        "journalctl",
        "--user",
        "-u",
        "nse-trader-ingest.service",
        "--since",
        since,
        "--until",
        f"{day} 23:59:59",
        "--no-pager",
        "-o",
        "short-iso",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
            env={**os.environ, "SYSTEMD_COLORS": "0"},
        )
    except FileNotFoundError:
        return {"day": day, "count": 0, "available": False, "error": "journalctl not found"}
    except subprocess.TimeoutExpired:
        return {"day": day, "count": 0, "available": False, "error": "journalctl timed out"}
    text = proc.stdout or ""
    count = count_journal_timeouts(text)
    return {
        "day": day,
        "count": count,
        "available": proc.returncode in (0, 1),
        "error": (proc.stderr or "").strip() if proc.returncode not in (0, 1) else None,
    }


def _systemctl_user(*args: str, timeout: float = 8.0) -> tuple[int, str, str]:
    cmd = ["systemctl", "--user", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "SYSTEMD_COLORS": "0"},
        )
    except FileNotFoundError:
        return 127, "", "systemctl not found"
    except subprocess.TimeoutExpired:
        return 124, "", "systemctl timed out"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def collect_unit_status(units: tuple[str, ...] = UNITS) -> dict:
    """Snapshot systemd --user unit state for agent triage."""
    units_out: dict[str, dict] = {}
    errors: list[str] = []

    for unit in units:
        code, out, err = _systemctl_user("show", unit, "--no-pager", *(f"--property={k}" for k in _PROP_KEYS))
        if code != 0:
            errors.append(f"{unit}: exit={code} {err or out}")
            units_out[unit] = {"ok": False, "error": err or out or f"exit {code}"}
            continue
        props: dict[str, str] = {}
        for line in out.splitlines():
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            props[key] = val
        units_out[unit] = {
            "ok": props.get("LoadState") == "loaded",
            "load_state": props.get("LoadState"),
            "active_state": props.get("ActiveState"),
            "sub_state": props.get("SubState"),
            "unit_file_state": props.get("UnitFileState"),
            "result": props.get("Result"),
            "exec_main_status": props.get("ExecMainStatus"),
            "last_trigger": props.get("LastTriggerUSec") or props.get("InactiveExitTimestamp"),
            "next_elapse": props.get("NextElapseUSecRealtime"),
            "fragment": props.get("FragmentPath"),
        }

    list_code, list_out, list_err = _systemctl_user("list-timers", "nse-trader*", "--no-pager", "--all")
    return {
        "available": list_code != 127,
        "units": units_out,
        "timers_table": list_out if list_code == 0 else "",
        "errors": errors + ([list_err] if list_code not in (0, 127) and list_err else []),
    }


def audit_systemd(units_info: dict) -> HealthReport:
    report = HealthReport()
    if not units_info.get("available"):
        report.add(
            Finding(
                check="systemd_available",
                severity=Severity.WARN,
                message="systemctl not available — skipped timer/service checks",
            )
        )
        return report

    units = units_info.get("units", {})
    missing = [u for u, info in units.items() if not info.get("ok")]
    if missing:
        report.add(
            Finding(
                check="systemd_units_loaded",
                severity=Severity.ERROR,
                message=f"Units not loaded / missing: {', '.join(missing)}",
                detail={"units": {u: units[u] for u in missing}},
            )
        )
    else:
        report.add(
            Finding(
                check="systemd_units_loaded",
                severity=Severity.OK,
                message=f"All {len(units)} nse-trader units loaded",
            )
        )

    for name, info in units.items():
        if not name.endswith(".timer"):
            continue
        state = (info.get("unit_file_state") or "").lower()
        active = info.get("active_state")
        if state and state not in {"enabled", "enabled-runtime", "static"}:
            report.add(
                Finding(
                    check="systemd_timer_enabled",
                    severity=Severity.WARN,
                    message=f"{name} unit_file_state={state!r} (expected enabled)",
                    detail=info,
                )
            )
        elif active not in {"active", "activating"}:
            report.add(
                Finding(
                    check="systemd_timer_active",
                    severity=Severity.WARN,
                    message=f"{name} active_state={active!r}",
                    detail=info,
                )
            )

    timeouts = units_info.get("ingest_timeouts") or {}
    n_to = int(timeouts.get("count") or 0) if timeouts else 0
    if n_to > 0:
        report.add(
            Finding(
                check="systemd_ingest_timeouts",
                severity=Severity.ERROR,
                message=(
                    f"nse-trader-ingest.service timed out {n_to} time(s) on "
                    f"{timeouts.get('day')} (last Result can still be success)"
                ),
                detail=timeouts,
            )
        )

    for name, info in units.items():
        if not name.endswith(".service"):
            continue
        result = (info.get("result") or "").lower()
        status = info.get("exec_main_status")
        # result=success or empty (never run) is fine; failure/exit-code is bad
        if result in {"failed", "exit-code", "signal", "core-dump", "timeout", "resources"}:
            report.add(
                Finding(
                    check="systemd_service_result",
                    severity=Severity.ERROR,
                    message=f"{name} last result={result!r} exec_main_status={status}",
                    detail=info,
                )
            )

    if not any(f.check.startswith("systemd_timer") for f in report.findings):
        report.add(
            Finding(
                check="systemd_timers",
                severity=Severity.OK,
                message="Timer enable/active state looks OK (or not yet triggered)",
            )
        )
    return report


def collect_artifacts(logs_dir: Path, *, day: str | None = None) -> dict:
    day = day or datetime.now(tz=IST).date().isoformat()
    eod_path = logs_dir / f"eod_{day}.json"
    health_path = logs_dir / "health.json"
    log_files = sorted(logs_dir.glob("*.log")) if logs_dir.exists() else []

    eod: dict | None = None
    if eod_path.exists():
        try:
            eod = json.loads(eod_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            eod = {"error": f"invalid json: {exc}"}

    return {
        "day": day,
        "eod_path": str(eod_path) if eod_path.exists() else None,
        "eod": eod,
        "health_path": str(health_path) if health_path.exists() else None,
        "log_files": [p.name for p in log_files],
        "log_tails": {
            p.name: _tail(p, n=8)
            for p in log_files
            if p.name in {"ingest.log", "paper.log", "eod.log"}
        },
    }


def _tail(path: Path, n: int = 8) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def audit_artifacts(artifacts: dict) -> HealthReport:
    report = HealthReport()
    day = artifacts["day"]
    if artifacts.get("eod"):
        eod = artifacts["eod"]
        if eod.get("error"):
            report.add(
                Finding(
                    check="eod_artifact",
                    severity=Severity.ERROR,
                    message=f"eod_{day}.json unreadable: {eod['error']}",
                )
            )
        elif eod.get("health_ok") is False:
            report.add(
                Finding(
                    check="eod_artifact",
                    severity=Severity.ERROR,
                    message=f"EOD for {day} recorded health_ok=false",
                    detail=eod,
                )
            )
        else:
            report.add(
                Finding(
                    check="eod_artifact",
                    severity=Severity.OK,
                    message=f"EOD artifact present for {day}",
                    detail={"n_trades": eod.get("n_trades"), "path": artifacts.get("eod_path")},
                )
            )
    else:
        report.add(
            Finding(
                check="eod_artifact",
                severity=Severity.WARN,
                message=f"No eod_{day}.json yet (normal before 15:45 IST)",
            )
        )

    if not artifacts.get("log_files"):
        report.add(
            Finding(
                check="service_logs",
                severity=Severity.WARN,
                message="No service *.log files yet",
            )
        )
    else:
        report.add(
            Finding(
                check="service_logs",
                severity=Severity.OK,
                message=f"Log files: {', '.join(artifacts['log_files'])}",
            )
        )

    report.merge(audit_bakeoff_metrics())
    return report


def audit_bakeoff_metrics() -> HealthReport:
    """Presence/staleness of offline/forward metrics tables (warn/info, not hard fail early)."""
    report = HealthReport()
    bake = ROOT / "data" / "store" / "experiments" / "meta_bakeoff"
    offline_daily = bake / "offline_daily.parquet"
    offline_metrics = bake / "offline_metrics.json"
    forward_daily = bake / "forward_daily.parquet"
    forward_metrics = bake / "forward_metrics.json"

    if offline_daily.exists() and not offline_metrics.exists():
        report.add(
            Finding(
                check="bakeoff_offline_metrics",
                severity=Severity.WARN,
                message="offline_daily exists but offline_metrics.json missing — run meta-bakeoff",
            )
        )
    elif offline_metrics.exists():
        try:
            payload = json.loads(offline_metrics.read_text(encoding="utf-8"))
            n = payload.get("n_rows")
            report.add(
                Finding(
                    check="bakeoff_offline_metrics",
                    severity=Severity.OK,
                    message=f"offline_metrics present (rows={n}, schema={payload.get('metrics_schema')})",
                    detail={"sparse": payload.get("sparse")},
                )
            )
        except json.JSONDecodeError as exc:
            report.add(
                Finding(
                    check="bakeoff_offline_metrics",
                    severity=Severity.WARN,
                    message=f"offline_metrics.json unreadable: {exc}",
                )
            )
    else:
        report.add(
            Finding(
                check="bakeoff_offline_metrics",
                severity=Severity.OK,
                message="No offline bake-off yet (optional until meta-bakeoff)",
            )
        )

    if forward_daily.exists():
        if not forward_metrics.exists():
            # Early week: warn only
            report.add(
                Finding(
                    check="bakeoff_forward_metrics",
                    severity=Severity.WARN,
                    message="forward_daily exists but forward_metrics.json missing — next EOD should refresh",
                )
            )
        else:
            try:
                stale = False
                if forward_daily.stat().st_mtime > forward_metrics.stat().st_mtime + 86400:
                    stale = True
                payload = json.loads(forward_metrics.read_text(encoding="utf-8"))
                n_days = 0
                for r in payload.get("rows") or []:
                    if r.get("block") == "policy":
                        n_days = max(n_days, int(r.get("n_days") or 0))
                sev = Severity.WARN if stale else Severity.OK
                msg = (
                    f"forward_metrics present (policy n_days≈{n_days}, sparse={payload.get('sparse')})"
                )
                if stale:
                    msg += " — stale vs forward_daily (>1d)"
                report.add(
                    Finding(
                        check="bakeoff_forward_metrics",
                        severity=sev,
                        message=msg,
                        detail={"n_days": n_days, "sparse": payload.get("sparse")},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                report.add(
                    Finding(
                        check="bakeoff_forward_metrics",
                        severity=Severity.WARN,
                        message=f"forward_metrics read failed: {exc}",
                    )
                )
    else:
        report.add(
            Finding(
                check="bakeoff_forward_metrics",
                severity=Severity.OK,
                message="No forward_daily yet (normal before first EOD bake-off day)",
            )
        )
    return report


def run_diagnose(*, day: str | None = None, cfg: MonitorConfig | None = None) -> dict:
    """
    One-shot agent triage bundle:
    health checks + systemd unit snapshot + today's EOD/log artifacts.
    """
    cfg = cfg or MonitorConfig.load()
    health = run_health_checks(cfg)
    systemd = collect_unit_status()
    systemd["ingest_timeouts"] = collect_ingest_timeouts(day=day)
    artifacts = collect_artifacts(cfg.logs_dir, day=day)

    extra = HealthReport()
    extra.merge(audit_systemd(systemd))
    extra.merge(audit_artifacts(artifacts))

    combined = HealthReport()
    combined.merge(health)
    combined.merge(extra)

    created = datetime.now(tz=IST).isoformat()
    hints = _fix_hints(combined)

    return {
        "ok": combined.ok,
        "created_at": created,
        "error_count": combined.error_count,
        "warn_count": combined.warn_count,
        "health": {
            "ok": health.ok,
            "error_count": health.error_count,
            "warn_count": health.warn_count,
            "findings": [_finding_dict(f) for f in health.findings],
        },
        "systemd": systemd,
        "artifacts": artifacts,
        "findings": [_finding_dict(f) for f in combined.findings],
        "fix_hints": hints,
        "agent_next_steps": [
            "Read findings[].detail for concrete files/lines",
            "Inspect data/logs/{ingest,paper,eod}.log tails in artifacts.log_tails",
            "systemctl --user status nse-trader-*.service",
            "python main.py test  # after code fix",
        ],
    }


def _finding_dict(f: Finding) -> dict:
    return {
        "check": f.check,
        "severity": f.severity.value,
        "message": f.message,
        "detail": f.detail,
    }


def _fix_hints(report: HealthReport) -> list[str]:
    hints: list[str] = []
    for f in report.findings:
        if f.severity != Severity.ERROR:
            continue
        if f.check == "log_errors":
            hints.append("Fix traceback/ERROR lines under data/logs/; re-run failed service")
        elif f.check == "systemd_ingest_timeouts":
            hints.append(
                "Ingest start-timeout at close — Fyers close hang; "
                "check TimeoutStartSec + close_fyers_bounded; journalctl --user -u nse-trader-ingest.service"
            )
        elif f.check == "systemd_service_result":
            unit = f.message.split()[0]
            hints.append(f"Inspect failed unit: systemctl --user status {unit}")
        elif f.check == "systemd_units_loaded":
            hints.append("Re-run deploy/enable-user-timers.sh to link/enable units")
        elif "allocator" in f.check:
            hints.append("Meta allocator may be stuck — check meta/allocator.py + allocation history")
        elif f.check.startswith("trade_"):
            hints.append("Trade log integrity issue — check experiments/paper.py strategy ids/clusters")
        elif f.check == "eod_artifact":
            hints.append("EOD health failed — open data/logs/eod_*.json and health.json")
        elif f.check == "bakeoff_offline_metrics":
            hints.append("python main.py meta-bakeoff  # rebuild offline_metrics")
        elif f.check == "bakeoff_forward_metrics":
            hints.append("Wait for EOD or: python main.py meta-status  # refreshes forward metrics")
    if not hints and not report.ok:
        hints.append("Review findings with severity=error")
    return list(dict.fromkeys(hints))


def format_diagnose(payload: dict) -> str:
    lines = [
        f"Diagnose: {'PASS' if payload['ok'] else 'FAIL'} "
        f"({payload['error_count']} error(s), {payload['warn_count']} warning(s))",
        f"Created: {payload['created_at']}",
        "",
        format_report(
            HealthReport(
                findings=[
                    Finding(
                        check=f["check"],
                        severity=Severity(f["severity"]),
                        message=f["message"],
                        detail=f.get("detail") or {},
                    )
                    for f in payload["findings"]
                ]
            )
        ),
    ]
    if payload.get("fix_hints"):
        lines.append("")
        lines.append("Fix hints:")
        for h in payload["fix_hints"]:
            lines.append(f"  - {h}")
    timers = (payload.get("systemd") or {}).get("timers_table") or ""
    if timers:
        lines.append("")
        lines.append("Timers:")
        lines.append(timers)
    return "\n".join(lines)


def write_diagnose_json(payload: dict, path: Path | None = None) -> Path:
    persistence = load_yaml("ops.yaml").get("persistence", {})
    out = path or (ROOT / persistence.get("logs_dir", "data/logs") / "diagnose.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # also refresh health.json for existing tooling
    write_report_json(
        HealthReport(
            findings=[
                Finding(
                    check=f["check"],
                    severity=Severity(f["severity"]),
                    message=f["message"],
                    detail=f.get("detail") or {},
                )
                for f in payload.get("health", {}).get("findings", [])
            ]
        ),
        out.parent / "health.json",
    )
    return out
