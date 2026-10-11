"""Monte Carlo simulation with realized (sampled) gank outcomes.

The exact solver works with expected gank losses because ganks do not affect
future states in Phase 1. The simulator instead samples every random event
(mixed actions, crash choices, gank Bernoulli draws) with a seeded NumPy
Generator, so realized payoff distributions are available for later
risk analysis (Phase 2). For a fixed seed the output is bit-for-bit
reproducible.

Farm collection is modelled as an expected (fractional) reward in Phase 1:
2 HP of minions at 90% collection efficiency yield 0.9 * 2 * farm_per_hp
resources deterministically. The simulator adds that expected amount; it does
not sample individual last hits. Plate income is deterministic as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .game import LaneGame, apply_choice
from .policy import Policy
from .state import State


@dataclass
class EpisodeLog:
    payoff: float                 # realized total payoff to A
    crash_resources: tuple
    ganks: tuple                  # number of successful ganks suffered by (A, B)
    actions: List[tuple]          # (a, b) per turn
    farm: tuple = (0.0, 0.0)      # farm collected (A, B)
    plates: tuple = (0.0, 0.0)    # plate income (A, B)


@dataclass
class SimulationResult:
    payoffs: np.ndarray           # realized payoff to A per episode
    ganks: np.ndarray             # shape (n, 2)
    crash_resources: np.ndarray   # shape (n, 2)
    farm: np.ndarray = None       # shape (n, 2)
    plates: np.ndarray = None     # shape (n, 2)

    @property
    def mean(self) -> float:
        return float(self.payoffs.mean())

    @property
    def stderr(self) -> float:
        return float(self.payoffs.std(ddof=1) / np.sqrt(len(self.payoffs)))

    def summary(self) -> dict:
        q = np.quantile(self.payoffs, [0.05, 0.25, 0.5, 0.75, 0.95])
        out = {
            "episodes": int(len(self.payoffs)),
            "mean": self.mean, "stderr": self.stderr,
            "std": float(self.payoffs.std(ddof=1)),
            "quantiles": dict(zip(["q05", "q25", "q50", "q75", "q95"], map(float, q))),
            "mean_ganks": self.ganks.mean(axis=0).tolist(),
            "mean_crash_resources": self.crash_resources.mean(axis=0).tolist(),
        }
        if self.farm is not None:
            out["mean_farm"] = self.farm.mean(axis=0).tolist()
        if self.plates is not None:
            out["mean_plates"] = self.plates.mean(axis=0).tolist()
        return out


def _sample(rng: np.random.Generator, p: np.ndarray) -> int:
    return int(rng.choice(len(p), p=p))


def play_episode(game: LaneGame, pol_a: Policy, pol_b: Policy, rng: np.random.Generator,
                 root: Optional[State] = None, k0: int = 0, record: bool = False) -> EpisodeLog:
    cfg = game.cfg
    s = game.initial_state() if root is None else root
    payoff = 0.0
    res = [0.0, 0.0]
    farm = [0.0, 0.0]
    plates = [0.0, 0.0]
    ganks = [0, 0]
    actions = []
    for k in range(k0, game.horizon):
        la, lb = game.legal(s, 0), game.legal(s, 1)
        a = la[_sample(rng, pol_a.move(k, s, 0, la))]
        b = lb[_sample(rng, pol_b.move(k, s, 1, lb))]
        if record:
            actions.append((a, b))
        tr = game.transition(s, k, a, b)
        # expected farm and plate income (deterministic given actions)
        payoff += tr.farm_payoff + tr.plate_payoff
        farm[0] += tr.farm[0]
        farm[1] += tr.farm[1]
        plates[0] += tr.plates[0]
        plates[1] += tr.plates[1]
        # realized gank checks (one per player per decision turn)
        ga = rng.random() < tr.gank_p[0]
        gb = rng.random() < tr.gank_p[1]
        ganks[0] += ga
        ganks[1] += gb
        payoff += cfg.gank_loss * (gb - ga)
        i = (_sample(rng, pol_a.choose(k, tr.pre_state, 0, tr.options_a, tr.options_b))
             if len(tr.options_a) > 1 else 0)
        j = (_sample(rng, pol_b.choose(k, tr.pre_state, 1, tr.options_b, tr.options_a))
             if len(tr.options_b) > 1 else 0)
        s, r = apply_choice(tr.pre_state, tr.options_a[i], tr.options_b[j], cfg)
        payoff += r
        res[0] += tr.options_a[i][0]
        res[1] += tr.options_b[j][0]
    return EpisodeLog(payoff, tuple(res), tuple(ganks), actions, tuple(farm), tuple(plates))


def simulate(game: LaneGame, pol_a: Policy, pol_b: Policy, episodes: int, seed: int,
             root: Optional[State] = None, k0: int = 0) -> SimulationResult:
    rng = np.random.default_rng(seed)
    pay = np.empty(episodes)
    g = np.empty((episodes, 2))
    cr = np.empty((episodes, 2))
    fm = np.empty((episodes, 2))
    pl = np.empty((episodes, 2))
    for e in range(episodes):
        log = play_episode(game, pol_a, pol_b, rng, root, k0)
        pay[e] = log.payoff
        g[e] = log.ganks
        cr[e] = log.crash_resources
        fm[e] = log.farm
        pl[e] = log.plates
    return SimulationResult(pay, g, cr, fm, pl)
