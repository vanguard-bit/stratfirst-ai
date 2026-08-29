from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.paper import reconcile_state, run_paper_day
from nse_trader.config import PortfolioConfig, ROOT, load_yaml
from ops.monitor.runner import format_report, run_health_checks

IST = ZoneInfo("Asia/Kolkata")


def run_forward_validation(
    days: int = 1,
    *,
    out_dir: Path | None = None,
    start_date: str | None = None,
) -> dict:
    """
    Forward validation loop — paper days + health audit.
    Designed for 6-month parallel run; each invocation advances one or more days.
    """
    cfg = PortfolioConfig.load()
    base = out_dir or (cfg.store_path / "experiments" / "forward")
    base.mkdir(parents=True, exist_ok=True)

    state_file = ROOT / load_yaml("ops.yaml")["persistence"]["state_dir"] / "portfolio.json"
    reconcile_state(state_file)

    if start_date:
        anchor = date.fromisoformat(start_date)
    else:
        anchor = datetime.now(tz=IST).date()

    day_results: list[dict] = []
    for offset in range(days):
        d = (anchor - timedelta(days=days - 1 - offset)).isoformat()
        paper = run_paper_day(d, out_dir=base / "paper")
        health = run_health_checks()
        day_results.append(
            {
                "date": d,
                "n_trades": paper["n_trades"],
                "health_ok": health.ok,
                "health_warnings": health.warn_count,
                "health_errors": health.error_count,
            }
        )

    manifest = {
        "validation_id": f"forward-{anchor.isoformat()}",
        "start_date": day_results[0]["date"] if day_results else anchor.isoformat(),
        "end_date": day_results[-1]["date"] if day_results else anchor.isoformat(),
        "days": days,
        "created_at": datetime.now(tz=IST).isoformat(),
        "days_detail": day_results,
        "health_last_ok": all(r["health_ok"] for r in day_results),
    }
    manifest_path = base / "forward_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
