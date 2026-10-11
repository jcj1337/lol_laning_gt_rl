"""Shared helpers of the RL experiments: output paths, scripted opponents, curve summaries, Markdown."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

from ...config import GameConfig
from ...io import dump_json

OUT = os.path.join("results", "rl")
CKPT = os.path.join(OUT, "checkpoints")
Z95 = 1.96
SCRIPTED = ["hold_farmer", "yield_farmer", "aggressive_trader", "safe_player", "shover", "conservative_recaller",
            "opportunistic_trader", "vision_player", "punisher", "hold", "trade", "yield", "safe"]


def scripted_agents(cfg: GameConfig, names: Sequence[str] = SCRIPTED) -> Dict[str, object]:
    from ...env import champion_agents
    agents = champion_agents(cfg)
    return {n: agents[n] for n in names}


def curve(history: List[dict], every: int = 1) -> List[List[float]]:
    """[(steps, mean training return)] downsampled; NaN rollouts (no finished episode) dropped."""
    return [[h["steps"], h["mean_return"]] for h in history[::every] if np.isfinite(h["mean_return"])]


def smooth_last(history: List[dict], n: int = 10) -> float:
    vals = [h["mean_return"] for h in history[-n:] if np.isfinite(h["mean_return"])]
    return float(np.mean(vals)) if vals else float("nan")


def mean_ci(values: Sequence[float]) -> str:
    """Mean over seeds +- 1.96 * standard error over seeds (t-free, few seeds: indicative only)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return f"{v.mean():+.2f}"
    return f"{v.mean():+.2f} ± {Z95 * v.std(ddof=1) / np.sqrt(len(v)):.2f}"


def ci(mean: float, se: float) -> str:
    return f"{mean:+.2f} ± {Z95 * se:.2f}"


def write(name: str, data: dict, md: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, f"{name}.json"))
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT}/{name}.json and .md")


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
