"""The simulation worker of the demo API (M6b; docs/m6/m6_plan.md §8.2).

A bounded bot game (Watch) takes 0.3 to 0.5 s of simulation, which would stall every human turn behind the single
simulation lock. It therefore runs in a separate *process*, started with the "spawn" context, that loads its own
registry and keeps its own copies of every module-level cache (the M5 loaders' networks, the belief library, the lane
caches). Nothing mutable is shared with the serving process: the only things that cross the boundary are the request
(two model ids and a turn count) and the finished game bundle. The functions below are the process's entry points;
`api.SimulationPool` owns the pool.
"""

from __future__ import annotations

import os
from typing import Optional

_SERVICE = None


def init_worker(root: str) -> None:
    """Load the registry in the worker process (`root` is the repository root the parent ran from, because the
    manifest's paths are relative to it)."""
    global _SERVICE
    os.chdir(root)
    from .service import DemoService
    _SERVICE = DemoService(max_sessions=1)


def ready() -> dict:
    if _SERVICE is None:
        return {"ready": False, "error": "the worker was not initialized"}
    return {"ready": _SERVICE.registry is not None, "error": _SERVICE.error}


def simulate(a: str, b: str, turns: int) -> dict:
    """One bounded bot game, generated in full (service.simulate), in this process."""
    from .service import ServiceError
    if _SERVICE is None or _SERVICE.registry is None:
        raise ServiceError(503, "engine_unavailable", (_SERVICE.error if _SERVICE else None) or "the worker has no engine")
    return _SERVICE.simulate(a, b, turns)


__all__ = ["init_worker", "ready", "simulate"]
