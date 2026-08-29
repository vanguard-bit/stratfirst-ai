from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class PortfolioConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> PortfolioConfig:
        return cls(raw=load_yaml("portfolio.yaml"))

    @property
    def total_capital(self) -> float:
        return float(self.raw["portfolio"]["total_capital"])

    @property
    def virtual_notional(self) -> float:
        return float(self.raw["virtual_books"]["per_strategy_notional"])

    @property
    def store_path(self) -> Path:
        rel = self.raw["data"]["store_path"]
        return ROOT / rel

    @property
    def fees_registry_path(self) -> Path:
        rel = self.raw["simulation"]["fees_registry"]
        return ROOT / rel
