from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nse_trader.config import PortfolioConfig, ROOT, load_yaml
from strategies.registry import load_enabled_strategies


@dataclass(frozen=True)
class DaySlice:
    start: int
    stop: int


def walk_forward_splits(
    n_days: int,
    train_pct: float = 0.7,
    n_folds: int = 3,
) -> list[tuple[DaySlice, DaySlice]]:
    """Expanding train window with sequential out-of-sample test chunks."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0 < train_pct < 1:
        raise ValueError("train_pct must be between 0 and 1")

    train_base = int(n_days * train_pct)
    test_total = n_days - train_base
    test_chunk = max(test_total // n_folds, 1)

    splits: list[tuple[DaySlice, DaySlice]] = []
    for fold in range(n_folds):
        test_start = train_base + fold * test_chunk
        if test_start >= n_days:
            break
        test_stop = min(test_start + test_chunk, n_days)
        splits.append((DaySlice(0, test_start), DaySlice(test_start, test_stop)))
    return splits


def _config_hash() -> str:
    parts: list[str] = []
    for name in ("portfolio.yaml", "strategies.yaml", "ops.yaml"):
        path = ROOT / "config" / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256("".join(parts).encode()).hexdigest()
    return digest[:16]


def run_backtest(run_id: str, out_dir: Path | None = None) -> dict:
    """
    Walk-forward backtest scaffold — writes manifest + split plan.
    Full bar replay lands in a later iteration; contract writes artifacts now.
    """
    cfg = PortfolioConfig.load()
    out = out_dir or (cfg.store_path / "experiments" / run_id)
    out.mkdir(parents=True, exist_ok=True)

    strategies = load_enabled_strategies()
    splits = walk_forward_splits(n_days=252, train_pct=0.7, n_folds=3)
    split_payload = [
        {
            "train": {"start": tr.start, "stop": tr.stop},
            "test": {"start": te.start, "stop": te.stop},
        }
        for tr, te in splits
    ]

    manifest = {
        "run_id": run_id,
        "config_hash": _config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategies": sorted(strategies.keys()),
        "n_splits": len(splits),
        "status": "complete",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "walk_forward.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    meta = load_yaml("strategies.yaml")
    (out / "config_snapshot.json").write_text(
        json.dumps({"strategies": meta.get("strategies", {})}, indent=2),
        encoding="utf-8",
    )
    return manifest
