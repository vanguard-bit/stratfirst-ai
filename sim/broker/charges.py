from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from sim.fees.sources.official import FeeRegistry

IST = ZoneInfo("Asia/Kolkata")
ADMIN_COMPONENT = "admin_square_off"
ADMIN_SEGMENT = "broker_admin"


def admin_square_off_charge(registry_path: Path) -> float:
    """Flat broker admin charge for MIS auto square-off (from fee registry)."""
    reg = FeeRegistry.load(registry_path)
    seg = reg.segments.get(ADMIN_SEGMENT)
    if not seg:
        raise KeyError(f"missing fee segment {ADMIN_SEGMENT!r} in registry")
    for comp in seg.components:
        if comp.name == ADMIN_COMPONENT and comp.rate_unit == "flat_inr":
            return float(comp.rate)
    raise KeyError(f"missing {ADMIN_COMPONENT!r} component in {ADMIN_SEGMENT}")
