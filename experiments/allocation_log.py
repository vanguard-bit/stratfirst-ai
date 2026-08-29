from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nse_trader.config import ROOT, load_yaml

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_PATH = ROOT / "data" / "state" / "meta_allocations.parquet"


def allocation_log_path() -> Path:
    raw = load_yaml("ops.yaml")
    mon = raw.get("monitoring", {})
    if mon.get("allocation_history"):
        return ROOT / mon["allocation_history"]
    state = raw.get("persistence", {}).get("state_dir", "data/state")
    return ROOT / state / "meta_allocations.parquet"


def log_allocation_snapshot(
    weights: dict[str, float],
    *,
    cluster_of: dict[str, str],
    regime: dict | None = None,
    mode: str = "rules",
    ts: datetime | None = None,
    path: Path | None = None,
) -> Path:
    """Append one meta-allocator rebalance row per strategy (for runtime audit)."""
    out = path or allocation_log_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = ts or datetime.now(tz=IST)
    regime = regime or {}

    rows = [
        {
            "ts": stamp.isoformat(),
            "strategy_id": sid,
            "cluster": cluster_of.get(sid, "?"),
            "weight": float(w),
            "mode": mode,
            "regime_adx": regime.get("adx"),
            "regime_vix": regime.get("vix"),
            "expiry_flag": regime.get("expiry_week", False),
        }
        for sid, w in weights.items()
        if w > 0
    ]
    new_df = pd.DataFrame(rows)
    if out.exists():
        existing = pd.read_parquet(out)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(out, index=False)
    return out
