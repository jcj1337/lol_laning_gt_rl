"""Own-perspective, fixed-length, Markov observations and legal-action masks.

`observe(state, k, side, cfg)` is a pure function of (k, state, side). Two agents that see the same
(k, state) from opposite seats receive each other's mirror image,

    observe(state, k, 1, cfg) == observe(state.mirror(), k, 0, cfg),

so one policy can play either seat, and an agent whose decision depends only on the observation is a
time-dependent Markov policy of the exact game (see agents.AgentPolicy).

Feature layout (every entry in [0, 1]; `obs_size(cfg)` entries in total):

  [0]              k / K
  [1]              (K - k) / K           turns remaining
  [2 : 2 + P]      one-hot of k mod P,   P = cfg.phase_period (spawn and creep phase of the lane dynamics)
  grid, own        for each cell x = -base .. +base in OWN coordinates: HP of own minions standing in the
                   cell (front packet plus rear packets, summed) / hp_cap
  grid, enemy      the same for enemy minions, in own coordinates (enemy own x -> -x)
  own side (5)     siege mass / siege_cap, vision turns / vision_turns, pending plates / plates_cap,
                   siege eligible (own crashed mass under the enemy tower, objectives enabled), has front
  enemy side (5)   the same five for the opponent

Champion mode (`cfg.champion_mode`) appends 2 x CHAMP_FEATURES entries, own champion then enemy champion:
one-hot of the champion's cell in OWN coordinates (2T + 1 cells; the enemy's cell c is shown at -c),
HP / champion_hp, absent / champion_absent_max, mechanical skill (seat A: skill_a, seat B: skill_b).
Then observe(lane, k, 1, cfg, champ) == observe(lane.mirror(), k, 0, swap_seats(cfg), champ.mirror()).
Legacy vectors are unchanged.

A grid value is clipped at 1 only if two packets of one side share a cell with more than hp_cap HP between
them (a capped front with a queued reinforcement at the same position), which the "block" cap rule permits
near the bases but which never happens at the lane. The legal-action mask is computed by the engine's own
`legal_actions`; there is no second legality rule.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import GameConfig
from ..mechanics import legal_actions
from ..state import State

SIDE_FEATURES = 5
CHAMP_FEATURE_TAIL = ("hp", "absent", "skill")
SIDE_FEATURE_NAMES = ("siege_mass", "vision", "plates_pending", "siege_eligible", "has_front")


def n_actions(cfg: GameConfig) -> int:
    """Size of the fixed action space: one slot per action code that can be legal somewhere in `cfg`
    (3 without positioning, 6 for the farming revision, 18 for the siege model)."""
    return max(cfg.action_codes()) + 1


def champ_features(cfg: GameConfig) -> int:
    """Champion features per side (champion mode): cell one-hot, HP, absence, skill."""
    return (2 * cfg.tower + 1) + len(CHAMP_FEATURE_TAIL)


def obs_size(cfg: GameConfig) -> int:
    n = 2 + cfg.phase_period + 2 * (2 * cfg.base + 1) + 2 * SIDE_FEATURES
    if cfg.champion_mode:
        n += 2 * champ_features(cfg)
    return n


def feature_names(cfg: GameConfig) -> List[str]:
    names = ["turn_fraction", "turns_remaining_fraction"]
    names += [f"phase_{p}" for p in range(cfg.phase_period)]
    names += [f"own_minions_x{x:+d}" for x in range(-cfg.base, cfg.base + 1)]
    names += [f"enemy_minions_x{x:+d}" for x in range(-cfg.base, cfg.base + 1)]
    names += [f"own_{n}" for n in SIDE_FEATURE_NAMES]
    names += [f"enemy_{n}" for n in SIDE_FEATURE_NAMES]
    if cfg.champion_mode:
        for who in ("own", "enemy"):
            names += [f"{who}_champ_x{x:+d}" for x in range(-cfg.tower, cfg.tower + 1)]
            names += [f"{who}_champ_{n}" for n in CHAMP_FEATURE_TAIL]
    assert len(names) == obs_size(cfg)
    return names


def _side_features(fx, fh, rears, siege, vision, plates, cfg: GameConfig, v: list, o: int) -> None:
    v[o] = siege / cfg.siege_cap
    v[o + 1] = vision / cfg.vision_turns if cfg.vision_turns > 0 else 0.0
    v[o + 2] = plates / cfg.plates_cap if cfg.plates_cap > 0 else 0.0
    v[o + 3] = 1.0 if (cfg.objectives and siege > 0) else 0.0
    v[o + 4] = 1.0 if fh > 0 else 0.0


def observe(state: State, k: int, side: int, cfg: GameConfig, champ=None) -> np.ndarray:
    """Own-perspective feature vector of side `side` at decision index k (float32, values in [0, 1]).
    Champion mode needs the champion state `champ` (lanegame.champions.ChampState)."""
    K = cfg.horizon
    P = cfg.phase_period
    base = cfg.base
    cells = 2 * base + 1
    cap = float(cfg.hp_cap)
    w = cfg.wave_hp / cap
    v = [0.0] * obs_size(cfg)              # a Python list is filled far faster than a float32 array
    v[0] = k / K
    v[1] = (K - k) / K
    v[2 + (k % P)] = 1.0
    own = state.side(side)
    enemy = state.side(1 - side)
    g0 = 2 + P                      # own grid
    g1 = g0 + cells                 # enemy grid
    fx, fh, rears = own[0], own[1], own[2]
    if fh > 0:
        v[g0 + fx + base] += fh / cap
    for x in rears:
        v[g0 + x + base] += w
    efx, efh, erears = enemy[0], enemy[1], enemy[2]
    if efh > 0:
        v[g1 - efx + base] += efh / cap
    for x in erears:
        v[g1 - x + base] += w
    for i in range(g0, g1 + cells):
        if v[i] > 1.0:              # co-located packets beyond the cap (possible near a base only)
            v[i] = 1.0
    s0 = g1 + cells
    _side_features(*own, cfg, v, s0)
    _side_features(*enemy, cfg, v, s0 + SIDE_FEATURES)
    if cfg.champion_mode:
        if champ is None:
            raise ValueError("champion mode: observe() needs the champion state")
        T = cfg.tower
        o = s0 + 2 * SIDE_FEATURES
        for who, sgn in ((side, 1), (1 - side, -1)):
            cs = champ.side(who)
            v[o + sgn * cs.pos + T] = 1.0
            o += 2 * T + 1
            v[o] = cs.hp / cfg.champion_hp
            v[o + 1] = cs.absent / cfg.champion_absent_max
            v[o + 2] = cfg.skill_a if who == 0 else cfg.skill_b
            o += len(CHAMP_FEATURE_TAIL)
    return np.asarray(v, dtype=np.float32)


def legal_set(state: State, side: int, cfg: GameConfig, champ=None):
    """The engine's legal action codes: mechanics.legal_actions, or in champion mode
    champions.champion_legal_actions (which builds on the lane's own wave and siege rules)."""
    if cfg.champion_mode:
        from ..champions import champion_legal_actions
        if champ is None:
            raise ValueError("champion mode: the legal set needs the champion state")
        return champion_legal_actions(state, champ, side, cfg)
    return legal_actions(state, side, cfg)


def legal_mask(state: State, side: int, cfg: GameConfig, n: int | None = None, champ=None) -> np.ndarray:
    """Boolean mask over the fixed action space; True exactly on the engine's legal set (`legal_set`)."""
    n = n_actions(cfg) if n is None else n
    m = np.zeros(n, dtype=bool)
    m[list(legal_set(state, side, cfg, champ))] = True
    return m
