"""Recordings of the M6 demo: the versioned app format, state digests and deterministic replay
(docs/m6/m6_plan.md §3 rules 5 and 8, §4.1 `Recording`; the contract is docs/m6/m6a_technical_report.md §4).

A recording is a pure function of (configuration, chance seed, both seats' codes). It holds:

    provenance   kind "research_record" (an M5 recording, re-verified), "new_simulation" (both seats bots, generated
                 by the app) or "human_game"; a non-research recording is an example, not evidence for M5's findings
    config       the configuration's name and SHA-256 (canonical JSON of GameConfig.to_dict())
    seeds        the chance seed, whether a visitor chose it, the policy seed and its rule. While a session is active
                 a drawn chance seed and the policy seed are withheld (a known seed would let anyone foresee the
                 chance rolls or compute a policy's sampled action offline); they are revealed when it stops
    seats        per seat: "human", or a registry model (id, file SHA-256, weights digest, sampled as M5 evaluated
                 it), or a scripted archetype
    turns        per turn t = 1..n: [code A, code B, digest of the state after turn t]
    beliefs      the opponent-modelling seats' posteriors after every position, only where the session may show them

`state_digest` is the first 16 hex digits of SHA-256 over the canonical JSON of (lane state, champion state, k).
`replay_frames` re-simulates the codes alone (no policy) in a fresh environment and checks every digest; it is what
history navigation and static replays use. `GameSession.rebuild` (session.py) re-simulates with the policies as well,
recomputing every policy seat's codes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import GameConfig

RECORD_SCHEMA = "lanegame.demo_recording"
RECORD_VERSION = 1
KINDS = ("research_record", "new_simulation", "human_game")
CONFIGS = {"m45_calibrated": GameConfig.m45_calibrated}
EXAMPLE_NOTE = ("One game, shown as an example. It is not evidence for or against M5's aggregate findings, which use "
                "all six seeds and many games per comparison.")


class ReplayMismatch(RuntimeError):
    """A re-simulation differs from its recording."""


def config_digest(cfg: GameConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_digest(lane, m4, k: int) -> str:
    blob = json.dumps([list(lane.to_list()), list(m4.to_list()), int(k)], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def config_of(rec: dict) -> GameConfig:
    name = rec["config"]["name"]
    if name not in CONFIGS:
        raise ValueError(f"unknown configuration {name!r}")
    cfg = CONFIGS[name]()
    if config_digest(cfg) != rec["config"]["digest"]:
        raise ReplayMismatch(f"{name}: the configuration's digest differs from the recording's")
    return cfg


def validate(rec: dict) -> None:
    """Structural checks of a recording (schema, version, kind, seats, turns)."""
    if rec.get("schema") != RECORD_SCHEMA or rec.get("version") != RECORD_VERSION:
        raise ValueError(f"not a {RECORD_SCHEMA} version {RECORD_VERSION} recording")
    if rec["provenance"]["kind"] not in KINDS:
        raise ValueError(f"unknown provenance kind {rec['provenance']['kind']!r}")
    if len(rec["seats"]) != 2:
        raise ValueError("a recording has two seats")
    for t in rec["turns"]:
        if len(t) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) for x in t[:2]):
            raise ValueError(f"malformed turn {t!r}")


def replay_frames(cfg: GameConfig, chance_seed: int, turns: Sequence[Sequence], p_from: int = 0,
                  p_to: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
    """Positions p_from..p_to and turns p_from+1..p_to of a recorded game, re-simulated from its codes in a fresh
    environment; every turn's state digest is checked (ReplayMismatch)."""
    from ..env import LaneEnv
    from . import frames as F
    n = len(turns)
    p_to = n if p_to is None else int(p_to)
    if not 0 <= p_from <= p_to <= n:
        raise ValueError(f"positions {p_from}..{p_to} outside 0..{n}")
    env = LaneEnv(cfg, record_ticks=True)
    env.reset(seed=int(chance_seed))
    positions, out_turns = [], []
    pre = F.position(cfg, env.state, env.champ, env.k, 0) if p_from == 0 else None
    if pre is not None:
        positions.append(pre)
    for i in range(p_to):
        a, b, digest = turns[i]
        want = i + 1 > p_from
        if want and pre is None:
            pre = F.position(cfg, env.state, env.champ, env.k, i)
            positions.append(pre)
        r = env.step(int(a), int(b))
        if state_digest(env.state, env.champ, env.k) != digest:
            raise ReplayMismatch(f"turn {i + 1}: the state digest differs from the recording")
        if want:
            post = F.position(cfg, env.state, env.champ, env.k, i + 1)
            out_turns.append(F.turn(cfg, r.info, pre, post))
            positions.append(post)
            pre = post
    if not positions:                                       # p_from == p_to > 0: one position, no turn
        positions.append(F.position(cfg, env.state, env.champ, env.k, p_to))
    return positions, out_turns


def replay_actions(cfg: GameConfig, chance_seed: int, actions: Sequence[Sequence[int]]) -> Tuple[List[list], float]:
    """Turns [a, b, digest] and seat A's total reward of an action list (a v1 viewer recording)."""
    from ..env import LaneEnv
    env = LaneEnv(cfg)
    env.reset(seed=int(chance_seed))
    out, total = [], 0.0
    for a, b in actions:
        r = env.step(int(a), int(b))
        total += r.rewards[0]
        out.append([int(a), int(b), state_digest(env.state, env.champ, env.k)])
    return out, total


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


__all__ = ["RECORD_SCHEMA", "RECORD_VERSION", "KINDS", "EXAMPLE_NOTE", "ReplayMismatch", "config_digest",
           "state_digest", "config_of", "validate", "replay_frames", "replay_actions", "file_sha256"]
