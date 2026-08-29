from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sim.fees.sources.official import FeeComponent, FeeRegistry, SegmentFees
from nse_trader.config import ROOT

SEED_PATH = ROOT / "config" / "fees_official_seed.json"


def load_seed_registry(broker_profile: str = "zerodha") -> FeeRegistry:
    """Load cited official snapshot — always works offline."""
    data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    fetched_at = datetime.now(timezone.utc).isoformat()
    url_by_id = {s["id"]: s["url"] for s in data.get("sources", [])}

    reg = FeeRegistry(
        version=data.get("version", 1),
        updated_at=fetched_at,
        broker_profile=broker_profile,
        sources=[{"id": "official_seed", "path": str(SEED_PATH), **s} for s in data.get("sources", [])],
    )
    for key, seg in data["segments"].items():
        components = []
        for c in seg["components"]:
            src_id = c.get("source_id", "zerodha_charges")
            components.append(
                FeeComponent(
                    name=c["name"],
                    rate=c["rate"],
                    rate_unit=c["rate_unit"],
                    side=c["side"],
                    source_url=url_by_id.get(src_id, str(SEED_PATH)),
                    source_label=c["source_label"],
                    fetched_at=fetched_at,
                    raw_text=c["source_label"],
                )
            )
        reg.segments[key] = SegmentFees(
            segment=seg["segment"],
            exchange=seg["exchange"],
            components=components,
        )
    reg.sources.append({"id": "seed_mode", "offline": True})
    return reg
