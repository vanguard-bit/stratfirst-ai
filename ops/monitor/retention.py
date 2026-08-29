from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nse_trader.config import ROOT, load_yaml
from ops.monitor.models import Finding, HealthReport, Severity

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class RetentionPolicy:
    rotate_after_days: int = 180
    auto_rotate: bool = False
    reminder_enabled: bool = True
    reminder_state_file: Path = Path("data/state/retention_reminder.json")
    reminder_banner_file: Path = Path("data/state/ROTATE_WHEN_READY.txt")
    repeat_every_days: int = 7

    @classmethod
    def load(cls) -> RetentionPolicy:
        raw = load_yaml("ops.yaml")
        ret = raw.get("retention", {})
        reminder = ret.get("reminder", {})
        return cls(
            rotate_after_days=int(ret.get("rotate_after_days", 180)),
            auto_rotate=bool(ret.get("auto_rotate", False)),
            reminder_enabled=bool(reminder.get("enabled", True)),
            reminder_state_file=ROOT / reminder.get("state_file", "data/state/retention_reminder.json"),
            reminder_banner_file=ROOT / reminder.get("banner_file", "data/state/ROTATE_WHEN_READY.txt"),
            repeat_every_days=int(reminder.get("repeat_every_days", 7)),
        )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _oldest_data_date(store_dir: Path, logs_dir: Path, state_dir: Path) -> date | None:
    oldest_ts: float | None = None
    skip_names = {"retention_reminder.json", "ROTATE_WHEN_READY.txt"}

    for base in (store_dir, logs_dir, state_dir):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.name in skip_names:
                continue
            ts = path.stat().st_mtime
            oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)

    if oldest_ts is None:
        return None
    return datetime.fromtimestamp(oldest_ts, tz=IST).date()


def _load_reminder_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_reminder_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_banner(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_banner(path: Path) -> None:
    if path.exists():
        path.unlink()


def check_retention_reminder(
    store_dir: Path,
    logs_dir: Path,
    state_dir: Path,
    policy: RetentionPolicy | None = None,
    *,
    today: date | None = None,
) -> HealthReport:
    """Warn when data age exceeds threshold. Never deletes or archives anything."""
    policy = policy or RetentionPolicy.load()
    report = HealthReport()
    today = today or datetime.now(tz=IST).date()

    if policy.auto_rotate:
        report.add(
            Finding(
                check="retention_policy",
                severity=Severity.ERROR,
                message="auto_rotate must stay false — rotation is manual only",
            )
        )
        return report

    first_date = _oldest_data_date(store_dir, logs_dir, state_dir)
    if first_date is None:
        report.add(
            Finding(
                check="retention_age",
                severity=Severity.OK,
                message="No runtime data yet — rotation reminder starts after first ingest",
                detail={"rotate_after_days": policy.rotate_after_days, "auto_rotate": False},
            )
        )
        _clear_banner(policy.reminder_banner_file)
        return report

    age_days = (today - first_date).days
    detail = {
        "first_data_date": first_date.isoformat(),
        "age_days": age_days,
        "rotate_after_days": policy.rotate_after_days,
        "auto_rotate": False,
    }

    if age_days < policy.rotate_after_days:
        report.add(
            Finding(
                check="retention_age",
                severity=Severity.OK,
                message=(
                    f"Data age {age_days}d — rotation reminder at {policy.rotate_after_days}d "
                    f"(nothing auto-deleted)"
                ),
                detail=detail,
            )
        )
        _clear_banner(policy.reminder_banner_file)
        return report

    # Past threshold — data stays; remind operator to rotate manually.
    if not policy.reminder_enabled:
        report.add(
            Finding(
                check="retention_reminder",
                severity=Severity.WARN,
                message=(
                    f"Data is {age_days}d old (>{policy.rotate_after_days}d). "
                    "Manual rotation recommended — auto_rotate is off, nothing deleted."
                ),
                detail=detail,
            )
        )
        return report

    state = _load_reminder_state(policy.reminder_state_file)
    last_reminder_raw = state.get("last_reminder_at")
    last_reminder = date.fromisoformat(last_reminder_raw) if last_reminder_raw else None
    due_since_raw = state.get("rotation_due_since")
    due_since = date.fromisoformat(due_since_raw) if due_since_raw else first_date

    should_refresh_banner = (
        last_reminder is None
        or (today - last_reminder).days >= policy.repeat_every_days
    )

    if should_refresh_banner:
        state.update(
            {
                "first_data_at": first_date.isoformat(),
                "rotation_due_since": due_since.isoformat(),
                "last_reminder_at": today.isoformat(),
                "reminder_count": int(state.get("reminder_count", 0)) + 1,
                "auto_rotate": False,
            }
        )
        _save_reminder_state(policy.reminder_state_file, state)
        banner = [
            "NSE Trader — manual rotation reminder",
            "========================================",
            f"First data:     {first_date.isoformat()}",
            f"Data age:       {age_days} days",
            f"Threshold:      {policy.rotate_after_days} days (6 months)",
            "Auto-rotate:    OFF — no files were deleted or moved",
            "",
            "When you are ready, manually archive old logs / compress cold parquet.",
            "Config: config/ops.yaml → retention",
            f"State:  {_display_path(policy.reminder_state_file)}",
        ]
        _write_banner(policy.reminder_banner_file, banner)

    report.add(
        Finding(
            check="retention_reminder",
            severity=Severity.WARN,
            message=(
                f"Data is {age_days}d old (>{policy.rotate_after_days}d). "
                f"Review {_display_path(policy.reminder_banner_file)} — "
                "rotate manually when ready; nothing auto-deleted."
            ),
            detail={
                **detail,
                "rotation_due_since": due_since.isoformat(),
                "reminder_count": state.get("reminder_count", 1),
                "banner_file": _display_path(policy.reminder_banner_file),
            },
        )
    )
    return report
