"""Plain-text rendering of lane traces (global coordinates, A pushes toward +)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import GameConfig
from .mechanics import step_turn
from .state import State


def lane_row(A: Sequence, B: Sequence, siege: Sequence[int], cfg: GameConfig, width: int = 6) -> str:
    cells = {x: "" for x in range(-cfg.base, cfg.base + 1)}
    for x, h in A:
        cells[x] += f"a{h}"
    for x, h in B:
        cells[-x] += f"b{h}"
    if siege[0]:
        cells[cfg.tower] += f"A*{siege[0]}"   # crashed A minions: packets ("packet" mode) or HP ("mass")
    if siege[1]:
        cells[-cfg.tower] += f"B*{siege[1]}"
    return " ".join(f"{(c or '.'):>{width}}" for _, c in sorted(cells.items()))


def header(cfg: GameConfig, width: int = 6) -> str:
    return " ".join(f"{x:>{width}}" for x in range(-cfg.base, cfg.base + 1))


def state_row(s: State, cfg: GameConfig) -> str:
    A = ([(s.ax, s.ah)] if s.ah else []) + [(x, cfg.wave_hp) for x in s.ar]
    B = ([(s.bx, s.bh)] if s.bh else []) + [(x, cfg.wave_hp) for x in s.br]
    return lane_row(A, B, (s.asg, s.bsg), cfg)


def _side_note(s: State, cfg: GameConfig) -> str:
    note = f"   vision A{s.va} B{s.vb}"
    if cfg.objectives:
        note += f" plates A{s.ap} B{s.bp}"
    return note


def trace_plan(cfg: GameConfig, s: State, k0: int, plan: Sequence[Tuple[int, int]]) -> Tuple[List[str], List]:
    """Run an action plan from (k0, s); return printable lines and the TurnResults."""
    w = (24 if cfg.objectives else 18) if cfg.positioning else 14
    lines = [f"{'':{w}}" + header(cfg),
             f"{'start k=' + str(k0):{w}}" + state_row(s, cfg) + _side_note(s, cfg)]
    results = []
    for i, (a, b) in enumerate(plan):
        k = k0 + i
        r = step_turn(s, k, a, b, cfg, record=True)
        for tr in r.trace:
            tag = f"k{k} t{tr['tick'] + 1} {cfg.label(a)}/{cfg.label(b)}"
            note = ""
            if tr["fight"]:
                note += f"  fight: A-{tr['fight'][4]} B-{tr['fight'][5]}"
            if any(tr["crashed"]):
                note += f"  CRASH A{tr['crashed'][0]} B{tr['crashed'][1]}"
            if any(tr.get("thinned", (0, 0))):
                note += f"  thin A{tr['thinned'][0]} B{tr['thinned'][1]}"
            lines.append(f"{tag:{w}}" + lane_row(tr["A"], tr["B"], tr["siege"], cfg) + note)
        end = f"{'  end of turn':{w}}" + state_row(r.state, cfg) + _side_note(r.state, cfg)
        if any(r.plates):
            end += f"  plates claimed A{r.plates[0]} B{r.plates[1]}"
        lines.append(end)
        results.append(r)
        s = r.state
    return lines, results
