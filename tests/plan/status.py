"""Track which plan phases are implemented — flip True as code lands."""

from __future__ import annotations

import importlib.util


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


PHASE_IMPLEMENTED: dict[str, bool] = {
    "phase0": True,
    "phase1": _has_module("sim.orders"),
    "phase2": _has_module("sim.exchange.rules"),
    "phase3": _has_module("sim.broker.square_off"),
    "phase4": _has_module("sim.pipeline"),
    "phase5": _has_module("data.ingest.store"),
    "phase6": _has_module("strategies.cluster_a.momentum"),
    "phase6b": _has_module("features.llm_gemini"),
    "phase7": _has_module("experiments.backtest"),
    "phase8": _has_module("experiments.paper"),
}


def phase_ready(phase: str) -> bool:
    return PHASE_IMPLEMENTED.get(phase, False)
